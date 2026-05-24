import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

from data_loader import prepare_data, ESM_MODEL
from model import ESMClassifier

# ── Hyperparameters ────────────────────────────────────────────────────────────
FREEZE_LAYERS = 6    # freeze first 6 of 12 ESM layers; fine-tune last 6 + head
BATCH_SIZE    = 16   # smaller than custom transformer — ESM sequences use more memory
EPOCHS        = 50
PATIENCE      = 10   # early stopping patience (tracked by val accuracy)
CHECKPOINT    = "best_esm_model.pt"


def main():
    # 1. Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, master_label_map, class_weights = prepare_data(
        batch_size=BATCH_SIZE
    )
    NUM_CLASSES = len(master_label_map)

    # 2. Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 3. Model ─────────────────────────────────────────────────────────────────
    model = ESMClassifier(ESM_MODEL, NUM_CLASSES, freeze_layers=FREEZE_LAYERS)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}  ({100 * trainable / total:.1f}%)")

    # 4. Loss + Optimizer ──────────────────────────────────────────────────────
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion      = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)

    # Differential learning rates:
    #   backbone → 3e-5  (pre-trained; nudge gently)
    #   head     → 1e-3  (randomly initialised; train normally)
    optimizer = optim.AdamW([
        {'params': model.esm.parameters(),        'lr': 3e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[3e-5, 1e-3],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=0.1,   # 10% warmup
    )

    # 5. Training loop ─────────────────────────────────────────────────────────
    best_val_acc     = 0.0
    patience_counter = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for input_ids, attention_mask, labels in bar:
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(input_ids, attention_mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in val_loader:
                input_ids      = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels         = labels.to(device)

                outputs   = model(input_ids, attention_mask)
                val_loss += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = (correct / total) * 100

        print(f"\n=> Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")

        # ── Checkpoint + early stopping (tracked by val accuracy) ──────────
        if val_accuracy > best_val_acc:
            best_val_acc     = val_accuracy
            patience_counter = 0
            torch.save({
                'epoch':            epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_accuracy':     best_val_acc,
                'master_label_map': master_label_map,
                'esm_model':        ESM_MODEL,
                'freeze_layers':    FREEZE_LAYERS,
                'num_classes':      NUM_CLASSES,
            }, CHECKPOINT)
            print(f"   ✓ Best model saved  (val acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}.")
                break

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
