# GNN-Based Intrusion Detection System: FLASH vs MAGIC

This project implements and compares two Graph Neural Network (GNN) approaches for provenance-based intrusion detection: **FLASH** and **MAGIC**. It provides a complete pipeline for loading provenance graphs, training node-level classifiers, cross-validating models, evaluating with rich metrics, and generating visual comparisons.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Documentation](#documentation)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
Provenance graphs capture system events (processes, files, sockets) and their relationships. This project trains GNNs to detect malicious nodes by learning from graph structure and node/edge features. Two architectures are implemented and compared:

- **FLASH**: Uses mean aggregation in message passing (GCN-style) for node representation learning.
- **MAGIC**: Uses attention-based aggregation to weight neighbor contributions.

## Dataset
The dataset contains DGL heterogeneous graphs stored as pickle files in `data/`:

- `train0.pkl`, `train1.pkl`, `train2.pkl`, `train3.pkl` - Training graphs
- `test0.pkl` - Test graph
- `metadata.json` - Dataset metadata (feature dimensions, counts, malicious labels)

Graph structure (from inspection):
- Node features: `attr` tensor of shape `[num_nodes, 5]`
- Edge features: `attr` tensor of shape `[num_edges, 17]`
- Node types: 5 types (0-4), with type 4 representing rare/malicious entities
- Labels: Nodes with `type == 4` are treated as positive (malicious)

## Project Structure
```
gnn-ids-project/
├── data/                    # Dataset files (.pkl, metadata.json)
├── docs/                    # Extracted paper texts (flash.txt, usenixsecurity24-jia-zian.txt)
├── out/                     # Outputs (checkpoints, plots, reports)
├── src/                     # Source code
│   ├── data.py              # Basic data loading
│   ├── data_v2.py           # Improved data utilities
│   ├── flash.py             # FLASH model (basic)
│   ├── flash_full.py        # FLASH model (full implementation)
│   ├── magic.py             # MAGIC model (basic)
│   ├── magic_full.py        # MAGIC model (full implementation)
│   ├── models.py            # Legacy models
│   ├── train.py             # Basic training
│   ├── train_full.py        # Full training with cross-validation
│   ├── eval.py              # Basic evaluation
│   ├── eval_full.py         # Full evaluation with rich metrics
│   ├── report.py            # Generate plots and reports
│   └── compare.py           # Comparison utilities
├── analysis.ipynb           # Jupyter notebook for analysis
├── COMPARISON_REPORT.md     # Detailed comparison results
├── FINAL_REPORT.md          # Final project report
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── flash.pdf, usenixsecurity24-jia-zian.pdf  # Reference papers
```

## Installation

### Prerequisites
- Python 3.10
- Virtual environment recommended

### Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Key packages:
- `torch==2.2.2` - Deep learning framework
- `dgl==2.2.0` - Graph neural network library
- `numpy==1.26.4` - Numerical computing
- `scikit-learn` - Machine learning utilities and metrics
- `matplotlib`, `seaborn` - Visualization
- `pypdf` - PDF parsing for papers
- `gensim` - Word embeddings (for future semantic encoding)

## Usage

### 1. Train Models
Train both models with cross-validation and save checkpoints:

```bash
python src/train_full.py
```

This will:
- Perform 4-fold cross-validation on training graphs
- Train final models on all training data
- Save checkpoints to `out/FLASH_full.pt` and `out/MAGIC_full.pt`

### 2. Evaluate Models
Evaluate trained models on the test set with comprehensive metrics:

```bash
python src/eval_full.py
```

Outputs include:
- Precision, Recall, F1-score
- ROC-AUC and PR-AUC
- Confusion matrix (TP, FP, TN, FN)

### 3. Generate Reports and Plots
Create ROC and Precision-Recall curves:

```bash
python src/report.py
```

Generates `out/FLASH_roc.png`, `out/FLASH_pr.png`, `out/MAGIC_roc.png`, `out/MAGIC_pr.png`.

### 4. Interactive Analysis
Use the Jupyter notebook for exploration:

```bash
jupyter notebook analysis.ipynb
```

## Models

### FLASH
```python
class FlashFull(nn.Module):
    def __init__(self, in_dim=5, hid=128, layers=3):
        # Multi-layer GNN with mean aggregation
        ...
    def forward(self, g, x):
        # Message passing with copy_u and mean aggregation
        ...
```

### MAGIC
```python
class MagicFull(nn.Module):
    def __init__(self, in_dim=5, hid=128, layers=3):
        # Multi-layer GNN with attention
        self.attn = nn.Linear(hid, 1)
        ...
    def forward(self, g, x):
        # Attention-weighted aggregation
        g.ndata['a'] = self.attn(h)
        g.update_all(dgl.function.u_mul_v('h','a','m'), dgl.function.mean('m','h'))
        ...
```

Both models use a linear classifier head for node-level binary classification.

## Evaluation

Metrics computed on test set:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **ROC-AUC**: Area under ROC curve
- **PR-AUC**: Area under Precision-Recall curve
- **Confusion Matrix**: TP, FP, TN, FN counts

## Results

Current evaluation results (on test0):
```
FLASH: P=0.0000 R=0.0000 F1=0.0000 AUC=0.8196 PR=0.0001 TP=0 FP=74 FN=3 TN=344690
MAGIC: P=0.0000 R=0.0000 F1=0.0000 AUC=0.1421 PR=0.0000 TP=0 FP=74 FN=3 TN=344690
```

*Note: Due to severe class imbalance (only 3 malicious nodes in test set) and the use of type==4 as proxy labels, standard metrics show challenges. FLASH achieves higher ROC-AUC but both models struggle with precision/recall on this label definition.*

Visualizations and detailed analysis available in:
- `COMPARISON_REPORT.md` - Detailed comparison
- `FINAL_REPORT.md` - Complete project report
- `out/FLASH_roc.png`, `out/MAGIC_roc.png`, `out/FLASH_pr.png`, `out/MAGIC_pr.png`

## Documentation

- **Papers**: `flash.pdf` (FLASH: A Comprehensive Approach to Intrusion Detection via Provenance Graph Representation Learning) and `usenixsecurity24-jia-zian.pdf` (MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning)
- **Extracted Text**: `docs/flash.txt`, `docs/usenixsecurity24-jia-zian.txt`
- **Team Design**: `GNN_IDS_Team_Design-1 (1).docx`, `Project Overview - Cybersecurity and Network System (3).docx`

## Future Improvements
- Implement faithful semantic encoding (Word2Vec) as described in FLASH
- Implement masked graph representation learning pretraining as in MAGIC
- Use accurate ground-truth labels from dataset
- Address class imbalance (oversampling, weighted loss, focal loss)
- Add more comprehensive hyperparameter tuning
- Implement ablation studies
- Add temporal features from edge types

## License
This project is for educational/research purposes.
