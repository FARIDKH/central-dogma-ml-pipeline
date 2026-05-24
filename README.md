# Protein Subcellular Localization — From Scratch Transformer to ESM-2 Transfer Learning

## Project Overview

An end-to-end deep learning pipeline for predicting **protein subcellular localization** — identifying where a protein resides within a cell (Cytoplasm, Nucleus, Mitochondria, etc.) based solely on its amino acid sequence.

The project follows a full research arc: starting with a custom Transformer trained from scratch, diagnosing its failure modes, and progressively improving it through architecture fixes, class-imbalance techniques, and finally transfer learning with Meta's **ESM-2** protein language model.

**Dataset:** [Protein Subcellular Localization](https://www.kaggle.com/datasets/lzyacht/proteinsubcellularlocalization) — 5,959 protein sequences across 11 imbalanced subcellular compartments.

---

## Results Summary

| Model | Accuracy | Macro F1 |
|---|---|---|
| Custom Transformer (buggy baseline) | 26% | 0.18 |
| Custom Transformer (fixed) | 61% | 0.51 |
| ESM-2 fine-tuned (v1) | 58% | 0.45 |
| ESM-2 fine-tuned (v2 — sampler fix) | 64% | 0.50 |
| **ESM-2 fine-tuned (v3 — final)** | **67%** | **0.53** |

---

## Stage 1 — Custom Transformer (From Scratch)

### Architecture

A standard Transformer encoder built in PyTorch:

- `nn.Embedding` (vocab_size=21, d_model=256) with `padding_idx=0`
- Sinusoidal Positional Encoding
- 6× `TransformerEncoderLayer` (GELU, 4 heads, dim_feedforward=1024)
- Learnable `[CLS]` token for sequence-level classification
- Classification head: `LayerNorm → Linear(256, 11)`

### Bugs Found and Fixed

Two critical bugs were discovered in the `forward()` method that caused a `RuntimeError: Mask size should match input size` and silently wrong outputs:

**Bug 1 — Mask shape mismatch (RuntimeError):**
After prepending the `[CLS]` token, the sequence length grew from `L` to `L+1`, but the padding mask was never extended to match. The fix was to prepend a `False` mask entry for the `[CLS]` token *before* passing to the encoder:
```python
# WRONG — mask is [batch, L], encoder input is [batch, L+1, d_model]
out = self.transformer_encoder(out, src_key_padding_mask=padding_mask)

# FIXED — extend mask first
cls_mask = torch.zeros(x.size(0), 1, dtype=torch.bool, device=x.device)
padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [batch, L+1]
out = self.transformer_encoder(out, src_key_padding_mask=padding_mask)
```

**Bug 2 — Contradictory readout (silent wrong output):**
The code performed masked average pooling to produce `[batch, d_model]`, then immediately indexed `out[:, 0]` which reduced it to `[batch]` — passing a scalar per sample into `LayerNorm` and `Linear`. The fix was to choose one readout strategy (CLS token) and remove the other:
```python
out = out[:, 0]  # CLS token → [batch, d_model]
```

### Key Training Techniques
- `AdamW` + `OneCycleLR` (10% warmup)
- `CrossEntropyLoss` with class weights (`compute_class_weight='balanced'`) + label smoothing 0.1
- Gradient clipping (`max_norm=1.0`)

---

## Stage 2 — ESM-2 Transfer Learning

### Why Transfer Learning

Even after fixing the bugs, the custom Transformer struggled on rare classes (vacuolar F1=0.28, peroxisomal F1=0.23) because it learned protein representations from scratch on only ~4,000 training sequences. ESM-2 was pre-trained by Meta on **250 million protein sequences**, giving it rich structural and functional embeddings out of the box.

### Model: `facebook/esm2_t12_35M_UR50D`

- 12 Transformer layers, hidden size 480, 35M parameters
- First 6 layers + embeddings frozen (preserve general protein representations)
- Last 6 layers fine-tuned (task-specific adaptation)
- Classification head added: `LayerNorm(480) → Dropout(0.3) → Linear(480, 11)`
- Sequence representation: `[CLS]` token at position 0

```
Trainable: ~16.8M / 33.5M params (50.3%)
```

### Techniques Applied

**Differential learning rates:**
Fine-tuned layers and the freshly initialised head need very different learning rates:
```python
optimizer = AdamW([
    {'params': model.esm.parameters(),        'lr': 3e-5},  # backbone — gentle
    {'params': model.classifier.parameters(), 'lr': 1e-3},  # head — normal
])
```

**WeightedRandomSampler:**
The dataset has a 22× imbalance (cytoplasmic=1411 vs vacuolar=63). Without intervention the model collapsed — cytoplasmic recall hit 93% while most other classes were ignored. `WeightedRandomSampler` forces every class to appear equally often in each batch:
```python
sample_weights = train_df['label'].map(lambda l: 1.0 / label_counts[l]).values
sampler = WeightedRandomSampler(weights=..., num_samples=len(train_dataset), replacement=True)
```

**Accuracy-based early stopping:**
With weighted CE + label smoothing, validation loss is a noisy proxy — the best accuracy epoch and best loss epoch diverged by up to 2%. Switching to tracking `val_accuracy` for checkpointing consistently saved a better model:
```python
if val_accuracy > best_val_acc:
    best_val_acc = val_accuracy
    torch.save(model.state_dict(), 'best_esm_model.pt')
```

**Increased dropout:**
Train loss (0.45) vs val loss (1.39) showed significant overfitting. Increasing classifier dropout from 0.1 → 0.3 reduced the gap and improved generalisation across medium-sized classes.

### Final Per-Class Results (ESM-2 v3)

```
                       precision    recall  f1-score   support

    chloroplast.fasta       0.73      0.84      0.78        68
    cytoplasmic.fasta       0.68      0.71      0.69       212
             ER.fasta       0.50      0.31      0.38        29
  extracellular.fasta       0.80      0.58      0.68       127
          Golgi.fasta       0.64      0.41      0.50        22
      lysosomal.fasta       0.29      0.47      0.36        15
  mitochondrial.fasta       0.65      0.72      0.69        76
        nuclear.fasta       0.64      0.76      0.69       125
    peroxisomal.fasta       0.44      0.17      0.24        24
plasma_membrane.fasta       0.77      0.76      0.77       186
       vacuolar.fasta       0.05      0.10      0.06        10

             accuracy                           0.67       894
            macro avg       0.56      0.53      0.53       894
         weighted avg       0.69      0.67      0.67       894
```

---

## Outlook

**What's working well:** The 7 larger classes (chloroplast, cytoplasmic, extracellular, mitochondrial, nuclear, plasma membrane, Golgi) all achieve F1 ≥ 0.50, with plasma membrane reaching 0.77.

**The hard ceiling:** The remaining weak classes (vacuolar=10, lysosomal=15, peroxisomal=24 test samples) are a data problem, not a model problem. No architecture or regularisation fix reliably learns from 10–44 training examples.

**To push further:**

| What | Expected gain |
|---|---|
| Larger ESM-2 (`esm2_t30_150M_UR50D`, 150M params) | +2–4% accuracy |
| More training data for rare classes | Largest potential gain |
| Focal Loss instead of weighted CE | Better handling of extreme imbalance |
| Multi-label approach (some proteins multi-localise) | Better biological accuracy |

---

## Tech Stack

| Category | Tools |
|---|---|
| Deep Learning | PyTorch |
| Transfer Learning | HuggingFace Transformers, ESM-2 |
| Data Processing | Pandas, NumPy, Scikit-Learn |
| Bioinformatics | Biopython |
| Hardware | CUDA (GPU-accelerated training) |

## Installation

```bash
python -m venv protein_env
source protein_env/bin/activate       # Mac/Linux
# protein_env\Scripts\activate        # Windows

pip install -r requirements.txt
```
