import os
import torch
import numpy as np
import pandas as pd
import kagglehub
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer

ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
MAX_LEN   = 510   # 512 total minus 2 special tokens (<cls> / <eos>)


class ProteinDatasetESM(Dataset):
    """
    Returns raw amino acid sequence strings and integer labels.
    The ESM tokenizer handles encoding inside the collate function,
    so no manual amino acid → integer mapping is needed here.
    """

    def __init__(self, dataframe: pd.DataFrame, label_to_int: dict, max_len: int = MAX_LEN):
        self.df           = dataframe.reset_index(drop=True)
        self.label_to_int = label_to_int
        self.max_len      = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        seq   = row['sequence'][:self.max_len]   # truncate before tokenizer adds special tokens
        label = self.label_to_int[row['label']]
        return seq, torch.tensor(label)


def make_collate_fn(tokenizer):
    """Returns a collate function bound to the given ESM tokenizer."""
    def esm_collate(batch):
        seqs   = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        encoded = tokenizer(
            seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        )
        return encoded['input_ids'], encoded['attention_mask'], labels
    return esm_collate


def prepare_data(batch_size: int = 16, max_len: int = MAX_LEN, esm_model: str = ESM_MODEL):
    """
    Downloads the dataset, splits it, and returns DataLoaders ready for ESM-2 fine-tuning.

    Returns:
        train_loader, val_loader, test_loader, master_label_map, class_weights
    """
    print("Downloading dataset...")
    path     = kagglehub.dataset_download("lzyacht/proteinsubcellularlocalization")
    csv_path = os.path.join(path, "1", "proteins.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(path, "proteins.csv")

    df = pd.read_csv(csv_path)

    # 70 / 15 / 15 stratified split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
    val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

    master_label_map = {label: i for i, label in enumerate(df['label'].unique())}

    # ESM tokenizer + collate
    tokenizer  = AutoTokenizer.from_pretrained(esm_model)
    collate_fn = make_collate_fn(tokenizer)

    # Datasets
    train_dataset = ProteinDatasetESM(train_df, master_label_map, max_len)
    val_dataset   = ProteinDatasetESM(val_df,   master_label_map, max_len)
    test_dataset  = ProteinDatasetESM(test_df,  master_label_map, max_len)

    # WeightedRandomSampler — equalise class frequency in every batch
    # Fixes the majority-class collapse problem (cytoplasmic=1411 vs vacuolar=63)
    label_counts   = train_df['label'].value_counts()
    sample_weights = train_df['label'].map(lambda l: 1.0 / label_counts[l]).values
    sampler        = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(train_dataset),
        replacement=True,
    )

    # sampler replaces shuffle=True for train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,    collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,    collate_fn=collate_fn)

    # Class weights for weighted cross-entropy loss
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_df['label']),
        y=train_df['label'],
    )

    print(f"Classes: {len(master_label_map)}  |  Train: {len(train_dataset)}  Val: {len(val_dataset)}  Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader, master_label_map, class_weights
