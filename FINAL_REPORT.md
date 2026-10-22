# Final Project Report: GNN-Based IDS Comparison (FLASH vs MAGIC)

## Summary
This project compares two GNN-based intrusion detection approaches on provenance graphs: FLASH and MAGIC. We implement baseline and extended models, add cross-validation, richer metrics, visualizations, and documentation.

## Methodology
- Dataset: DGL heterographs (train0-3, test0) with node attr (5) and edge attr (17).
- Labels: nodes with type==4 treated as malicious (rare class).
- Models: multi-layer GNN encoders with mean (FLASH) and attention-based (MAGIC) aggregation, linear node classifier.
- Training: 4-fold cross-validation on train set, final model trained on all train graphs.
- Evaluation: node-level classification on test0; metrics include Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix.

## Results
Results saved to out/eval_report.txt and plots (out/*_roc.png, out/*_pr.png). Check out/ for latest metrics.

## Repository Structure
- src/: data loaders, model implementations (flash/magic variants), train/eval/report scripts.
- data/: pickle graphs and metadata.json.
- docs/: extracted paper text for reference.
- out/: checkpoints, metrics, and plots.
- README.md, COMPARISON_REPORT.md, FINAL_REPORT.md.

## Usage
```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/train_full.py
python src/eval_full.py
python src/report.py
```

## Conclusion
The pipeline supports reproducible comparison of FLASH and MAGIC on this dataset. Further work can refine semantic encoding and address class imbalance.
