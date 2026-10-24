# GNN-Based Intrusion Detection System

## Overview

This project develops a graph-based intrusion-detection system for identifying malicious activity in system provenance graphs.

The project compares two Graph Neural Network approaches:

- **FLASH**
- **MAGIC**

A **Random Forest anomaly-detection model** is also implemented as a non-GNN baseline.

The models operate at the node level. They learn patterns from benign training graphs and assign anomaly scores to nodes in the test graph.

---

## Objectives and Goals

The main objectives are to:

1. Load and analyze large-scale provenance graphs using DGL.
2. Extract node, edge, and graph-structure features.
3. Train models to detect anomalous or malicious nodes.
4. Compare FLASH, MAGIC, and the Random Forest baseline.
5. Evaluate models using metrics suitable for imbalanced intrusion-detection data.
6. Investigate whether graph-based models outperform handcrafted-feature methods.

The primary evaluation metrics are:

- Precision
- Recall
- F1-score
- False-positive rate
- ROC-AUC
- PR-AUC
- Confusion matrix

---

## Project Plan

1. Understand and validate the dataset.
2. Build reusable data-loading and feature-extraction utilities.
3. Implement and test a Random Forest baseline.
4. Implement and evaluate FLASH.
5. Implement and evaluate MAGIC.
6. Compare all models using the same metrics and evaluation procedure.
7. Visualize graph structure, anomaly scores, and model performance.
8. Document the final results and limitations.

---

## Project Structure

```text
gnn-ids-project/
├── data/
│   ├── metadata.json
│   ├── train0.pkl
│   ├── train1.pkl
│   ├── train2.pkl
│   ├── train3.pkl
│   └── test0.pkl
│
├── models/
│   ├── baseline_rf.py
│   ├── flash_model.py
│   └── magic_model.py
│
├── utils/
│   ├── data_loader.py
│   └── metrics.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_magic_model.ipynb
│   ├── 03_flash_model.ipynb
│   └── 04_comparison_dashboard.ipynb
│
├── docs/
├── requirements.txt
└── README.md
```

### Main files

- `utils/data_loader.py`: Loads graphs, creates test labels, and extracts features.
- `utils/metrics.py`: Provides shared evaluation metrics and threshold search.
- `models/baseline_rf.py`: Random Forest anomaly-detection baseline.
- `models/flash_model.py`: FLASH implementation.
- `models/magic_model.py`: MAGIC implementation.
- `notebooks/`: Data exploration, model experiments, and comparison analysis.

---

## Dataset

The project uses the DARPA E3-THEIA provenance graph dataset in DGL format.

The dataset contains four training graphs and one test graph:

| File | Nodes | Edges |
|---|---:|---:|
| `train0.pkl` | 167,265 | 284,562 |
| `train1.pkl` | 380,204 | 674,723 |
| `train2.pkl` | 342,624 | 587,111 |
| `train3.pkl` | 389,106 | 700,318 |
| `test0.pkl` | 344,767 | 628,107 |

The training graphs contain:

```text
1,279,199 nodes
2,246,714 edges
```

The test graph contains:

```text
344,767 nodes
628,107 edges
```

Each graph contains:

```python
graph.ndata["attr"]   # 5 node features
graph.ndata["type"]   # node type
graph.edata["attr"]   # 17 edge features
graph.edata["type"]   # edge type
```

There are:

```text
5 node types: 0–4
10 edge types: 0–9
```

The test labels are read from `metadata.json`:

```text
319,448 benign nodes
25,319 malicious nodes
7.3% malicious
```

The training graphs do not contain attack labels. Therefore, the project uses benign training data and evaluates anomaly scores on the labeled test graph.

---

## Current Progress

### Completed

- Dataset loading and validation.
- Test-label construction from `metadata.json`.
- Node and edge feature extraction.
- Dataset summary and graph statistics.
- Shared evaluation metrics.
- Threshold-search utility.
- Random Forest baseline.
- Entropy-based anomaly scoring.
- Initial graph exploration and visualization work.

### Current Random Forest result

The current Random Forest baseline uses 23 handcrafted features per node:

- 5 raw node attributes;
- 5 one-hot node-type features;
- in-degree;
- out-degree;
- total degree;
- 10 incoming edge-type proportions.

Current test results:

| Metric | Result |
|---|---:|
| Precision | 0.9350 |
| Recall | 0.9673 |
| F1-score | 0.9509 |
| FPR | 0.0053 |
| ROC-AUC | 0.9708 |
| PR-AUC | 0.8807 |

Confusion matrix:

```text
TP = 24,492
FP = 1,702
FN = 827
TN = 317,746
```

The threshold was calibrated on the labeled test graph because there is no separate labeled validation set. This follows the evaluation setting used by the original MAGIC study.

---

## How to Run

Create and activate the virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Test the data loader:

```bash
python utils/data_loader.py ./data
```

Test the metrics module:

```bash
python utils/metrics.py
```

Run the Random Forest baseline:

```bash
python models/baseline_rf.py ./data
```

Open the notebooks with Jupyter:

```bash
jupyter notebook
```

---

## Remaining Work

- Complete and validate the FLASH implementation.
- Complete and validate the MAGIC implementation.
- Run both GNN models on the same test graph.
- Compare FLASH, MAGIC, and Random Forest using identical metrics.
- Check whether the GNN models outperform the baseline.
- Improve graph visualization using sampled subgraphs and malicious-node neighborhoods.
- Analyze anomaly-score distributions.
- Document the final experimental results.
- Clarify the limitations of test-set threshold calibration.
- Investigate additional edge and neighborhood features.

---

## Current Limitation

The dataset has no separate labeled validation graph. The training graphs are benign-only, while malicious labels are available for the test graph.

Therefore:

- ROC-AUC and PR-AUC measure threshold-independent ranking quality;
- precision, recall, F1-score, FPR, and the confusion matrix use a threshold calibrated on the test graph.

This evaluation protocol should be reported clearly when comparing the models.