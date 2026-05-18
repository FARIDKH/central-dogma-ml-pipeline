# Bio-Informatics Pipeline: The Central Dogma & Protein Localization

## Project Overview
This project demonstrates an end-to-end Machine Learning pipeline for biological sequence analysis. It follows the **Central Dogma of Molecular Biology**: the process by which genetic information flows from DNA to RNA to functional Proteins.

The core objective is to predict **Subcellular Localization**—identifying where a protein will reside within a cell (e.g., Chloroplast, Nucleus, Cytoplasm) based solely on its primary amino acid sequence.

## Key Features
* **Sequence Translation:** Automated translation of genomic DNA sequences into protein sequences using `Biopython`.
* **Custom Neural Pipeline:** A PyTorch-based architecture designed for sequence modeling.
* **Dynamic Padding:** Implementation of a custom `collate_fn` to handle variable-length biological sequences efficiently during batching.
* **Representation Learning:** Utilizing embedding layers to map 20 standard amino acids into a continuous vector space for deep learning.

## Tech Stack
* **Deep Learning:** PyTorch
* **Bio-Informatics:** Biopython
* **Data Processing:** Pandas, NumPy, Scikit-Learn
* **Hardware Acceleration:** CUDA support for GPU-accelerated training.

## Installation & Setup

### 1. Environment Setup
It is recommended to use an isolated environment. You can use the provided `requirements.txt` to install all necessary dependencies:

```bash
# Create a virtual environment
python -m venv protein_env

# Activate it (Windows)
protein_env\Scripts\activate
# Activate it (Mac/Linux)
source protein_env/bin/activate

# Install dependencies
pip install -r requirements.txt