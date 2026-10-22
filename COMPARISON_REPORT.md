# GNN IDS Comparison Report (FLASH vs MAGIC)

## Data
- Graphs: 4 train (train0-3), 1 test (test0). Node features dim 5, edge features dim 17.
- Node labels: type == 4 marked malicious (from metadata/graph inspection).

## Models
- FLASH baseline and extended (mean aggregation over message passing).
- MAGIC baseline and extended (attention-weighted aggregation).

## Training
- Cross-validation on train graphs, final retrain on all train data.
- BCE loss on node-level predictions.

## Evaluation (test0)
See out/eval_full.py output and plots in out/:

- FLASH: ROC curve out/FLASH_roc.png, PR curve out/FLASH_pr.png
- MAGIC: ROC curve out/MAGIC_roc.png, PR curve out/MAGIC_pr.png

## Metrics
Generated via src/eval_full.py (P/R/F1/AUC/PR-AUC, confusion matrix).
Saved checkpoints: out/FLASH_full.pt, out/MAGIC_full.pt, out/MAGIC_full.pt, out/FLASH_full.pt (and earlier).

## Files created/modified
- src/flash_full.py, src/magic_full.py, src/train_full.py, src/eval_full.py, src/report.py, src/data_v2.py
- README.md, COMPARISON_REPORT.md
- docs/flash.txt, docs/usenixsecurity24-jia-zian.txt
- out/*.png, out/*_full.pt, out/eval_report.txt
