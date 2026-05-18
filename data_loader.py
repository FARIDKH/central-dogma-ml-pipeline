import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class ProteinDataset(Dataset):
    def __init__(self, csv_file):
        """
        Loads the Kaggle dataset and sets up the amino acid vocabulary.
        """
        # 1. Load the data
        # Assuming the CSV has columns: 'Sequence' and 'Subcellular_Localization'
        self.df = pd.read_csv(csv_file)
        
        # 2. Define the vocabulary (Dictionary Mapping)
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        self.aa_to_int = {aa: i+1 for i, aa in enumerate(self.amino_acids)}
        self.aa_to_int['<PAD>'] = 0 # 0 is reserved for padding
        
        # 3. Create a mapping for the target labels (e.g., Nucleus -> 0)
        self.unique_labels = self.df['Subcellular_Localization'].unique()
        self.label_to_int = {label: i for i, label in enumerate(self.unique_labels)}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Fetches one protein and converts it to integers.
        """
        row = self.df.iloc[idx]
        sequence = row['Sequence']
        label_str = row['Subcellular_Localization']
        
        # Convert protein string to list of integers
        # We use .get(char, 0) to handle any weird unknown characters by treating them as padding
        encoded_seq = [self.aa_to_int.get(char, 0) for char in sequence]
        
        # Convert to PyTorch tensors
        seq_tensor = torch.tensor(encoded_seq, dtype=torch.long)
        label_tensor = torch.tensor(self.label_to_int[label_str], dtype=torch.long)
        
        return seq_tensor, label_tensor

def pad_collate(batch):
    """
    This function tells the DataLoader how to handle a batch of proteins 
    that are all different lengths.
    """
    # Separate the sequences and the labels from the batch
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Pad the sequences to the length of the longest protein in THIS specific batch
    # batch_first=True makes the output shape [batch_size, max_length]
    # padding_value=0 uses our <PAD> token
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Stack the labels into a single tensor
    labels = torch.stack(labels)
    
    return padded_sequences, labels

# ==========================================
# How to use it in your training loop:
# ==========================================
if __name__ == "__main__":
    # 1. Instantiate the dataset
    # (Make sure you have the kaggle csv downloaded as 'protein_data.csv')
    dataset = ProteinDataset('protein_data.csv')
    
    # 2. Create the DataLoader
    # We pass our custom 'pad_collate' function here
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    
    # 3. Test it out!
    for batch_idx, (sequences, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print(f"Sequences Shape: {sequences.shape} -> [batch_size, max_sequence_length_in_batch]")
        print(f"Labels Shape: {labels.shape} -> [batch_size]")
        break # Just looking at the first batch