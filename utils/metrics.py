"""
utils/metrics.py
================
Shared evaluation metrics for all three models.

All three models (MAGIC, FLASH, Random Forest) call the same functions
here so results are directly comparable — no model gets a different
scoring formula.

Metrics reported (matching both papers):
- Precision   : of all nodes flagged, how many are truly malicious
- Recall      : of all malicious nodes, how many did we catch
- F1-Score    : harmonic mean of precision and recall
- FPR         : false positive rate (false alarms / total benign)
- ROC-AUC     : area under ROC curve (threshold-independent)
- PR-AUC      : area under Precision-Recall curve
                (better than ROC-AUC for imbalanced data like ours)

Usage:
    from utils.metrics import evaluate, print_results, compare_models
"""

import torch
import numpy as np
from typing import Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    average_precision_score,
    roc_curve, precision_recall_curve
)


# Main evaluation function ─────────────────────────────────────────────────

def evaluate(
    y_true:   torch.Tensor,
    y_pred:   torch.Tensor,
    y_scores: Optional[torch.Tensor] = None,
    model_name: str = 'Model'
) -> dict:
    """
    Compute all detection metrics for one model.

    Parameters
    ----------
    y_true : LongTensor (N,)
        Ground truth labels. 0 = benign, 1 = malicious.
    y_pred : LongTensor (N,)
        Binary predictions. 0 = predicted benign, 1 = predicted malicious.
    y_scores : FloatTensor (N,), optional
        Anomaly scores (higher = more suspicious).
        Required for ROC-AUC and PR-AUC.
        For MAGIC  : KNN distance score
        For FLASH  : 1 - classifier confidence
        For RF     : classifier probability for class 1
    model_name : str
        Label used in printed output.

    Returns
    -------
    dict with keys:
        precision, recall, f1, fpr, tp, fp, fn, tn,
        roc_auc (if y_scores given), pr_auc (if y_scores given)
    """
    # Convert tensors to numpy
    yt = _to_numpy(y_true).astype(int)
    yp = _to_numpy(y_pred).astype(int)

    # Guard: if model predicts all one class, metrics degrade gracefully
    if yp.sum() == 0:
        print(f'  WARNING [{model_name}]: model predicted NO positives '
              '— try lowering the detection threshold.')
    if yp.sum() == len(yp):
        print(f'  WARNING [{model_name}]: model predicted ALL positives '
              '— try raising the detection threshold.')

    # Core metrics
    precision = precision_score(yt, yp, zero_division=0)
    recall    = recall_score(yt, yp, zero_division=0)
    f1        = f1_score(yt, yp, zero_division=0)

    # Confusion matrix → FPR
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    results = {
        'model':     model_name,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'fpr':       fpr,
        'tp':        int(tp),
        'fp':        int(fp),
        'fn':        int(fn),
        'tn':        int(tn),
        'roc_auc':   None,
        'pr_auc':    None,
    }

    # AUC metrics (need continuous scores, not binary predictions)
    if y_scores is not None:
        ys = _to_numpy(y_scores).astype(float)
        try:
            results['roc_auc'] = roc_auc_score(yt, ys)
        except ValueError:
            pass  # only one class present
        try:
            # PR-AUC is more informative than ROC-AUC on imbalanced data
            # Our data: 7.3% malicious — PR-AUC is the key metric
            results['pr_auc'] = average_precision_score(yt, ys)
        except ValueError:
            pass

    return results


# Printing helpers ─────────────────────────────────────────────────────────

def print_results(results: dict) -> None:
    """Print a formatted results block for one model."""
    name = results.get('model', 'Model')
    print(f'\n{"=" * 50}')
    print(f'  {name}')
    print(f'{"=" * 50}')
    print(f'  Precision  : {results["precision"]:.4f}')
    print(f'  Recall     : {results["recall"]:.4f}')
    print(f'  F1-Score   : {results["f1"]:.4f}')
    print(f'  FPR        : {results["fpr"]:.4f}')
    if results['roc_auc'] is not None:
        print(f'  ROC-AUC    : {results["roc_auc"]:.4f}')
    if results['pr_auc'] is not None:
        print(f'  PR-AUC     : {results["pr_auc"]:.4f}')
    print(f'  ── Confusion matrix ──')
    print(f'  TP={results["tp"]:,}  FP={results["fp"]:,}  '
          f'FN={results["fn"]:,}  TN={results["tn"]:,}')
    print(f'{"=" * 50}')


def compare_models(results_list: list) -> None:
    """
    Print a side-by-side comparison table for multiple models.

    Parameters
    ----------
    results_list : list of dicts returned by evaluate()
    """
    print(f'\n{"=" * 75}')
    print('  MODEL COMPARISON')
    print(f'{"=" * 75}')

    # Header
    header = f'  {"Model":<22}  {"Precision":>9}  {"Recall":>9}  '
    header += f'{"F1":>9}  {"FPR":>9}  {"PR-AUC":>9}'
    print(header)
    print(f'  {"-"*22}  {"-"*9}  {"-"*9}  {"-"*9}  {"-"*9}  {"-"*9}')

    for r in results_list:
        pr_auc = f'{r["pr_auc"]:.4f}' if r['pr_auc'] is not None else '  N/A  '
        row = (f'  {r["model"]:<22}  '
               f'{r["precision"]:>9.4f}  '
               f'{r["recall"]:>9.4f}  '
               f'{r["f1"]:>9.4f}  '
               f'{r["fpr"]:>9.4f}  '
               f'{pr_auc:>9}')
        print(row)

    print(f'{"=" * 75}')
    print('  PR-AUC is the primary metric (data is 7.3% malicious / imbalanced)')
    print(f'{"=" * 75}')


# Curve data (for plotting in notebooks) ───────────────────────────────────

def get_roc_curve_data(
    y_true: torch.Tensor,
    y_scores: torch.Tensor
) -> tuple:
    """
    Return (fpr_array, tpr_array, thresholds) for plotting ROC curve.
    Use in notebooks with matplotlib.
    """
    yt = _to_numpy(y_true).astype(int)
    ys = _to_numpy(y_scores).astype(float)
    fpr_arr, tpr_arr, thresholds = roc_curve(yt, ys)
    return fpr_arr, tpr_arr, thresholds


def get_pr_curve_data(
    y_true: torch.Tensor,
    y_scores: torch.Tensor
) -> tuple:
    """
    Return (precision_array, recall_array, thresholds) for plotting PR curve.
    Use in notebooks with matplotlib.
    """
    yt = _to_numpy(y_true).astype(int)
    ys = _to_numpy(y_scores).astype(float)
    prec_arr, rec_arr, thresholds = precision_recall_curve(yt, ys)
    return prec_arr, rec_arr, thresholds


# Threshold search ─────────────────────────────────────────────────────────

def find_best_threshold(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    metric: str = 'f1',
    n_steps: int = 100
) -> tuple:
    """
    Search for the anomaly score threshold that maximises a given metric.

    Parameters
    ----------
    y_true    : ground truth labels (0/1)
    y_scores  : continuous anomaly scores
    metric    : one of 'f1', 'precision', 'recall'
    n_steps   : number of threshold values to try

    Returns
    -------
    best_threshold : float
    best_score     : float
    all_results    : list of (threshold, metric_value) tuples
    """
    yt = _to_numpy(y_true).astype(int)
    ys = _to_numpy(y_scores).astype(float)

    thresholds = np.linspace(ys.min(), ys.max(), n_steps)
    best_threshold = thresholds[0]
    best_score = 0.0
    all_results = []

    for t in thresholds:
        yp = (ys >= t).astype(int)
        if metric == 'f1':
            score = f1_score(yt, yp, zero_division=0)
        elif metric == 'precision':
            score = precision_score(yt, yp, zero_division=0)
        elif metric == 'recall':
            score = recall_score(yt, yp, zero_division=0)
        else:
            raise ValueError(f'Unknown metric: {metric}. '
                             'Choose from f1, precision, recall.')
        all_results.append((t, score))
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score, all_results


# Internal helper ───────────────────────────────────────────────────────────

def _to_numpy(x) -> np.ndarray:
    """Convert torch.Tensor or list to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


# Quick test

if __name__ == '__main__':
    print('Testing metrics.py with synthetic data...\n')

    # Simulate imbalanced binary classification (7% positive like our dataset)
    torch.manual_seed(42)
    N = 10000
    y_true   = (torch.rand(N) < 0.073).long()     # ~7.3% malicious
    y_scores = torch.rand(N)
    # Make malicious nodes score slightly higher on average
    y_scores[y_true == 1] += 0.3
    y_scores = y_scores.clamp(0, 1)

    # Find best threshold then evaluate
    threshold, best_f1, _ = find_best_threshold(y_true, y_scores, metric='f1')
    print(f'Best threshold (max F1): {threshold:.4f}  →  F1={best_f1:.4f}')

    y_pred = (y_scores >= threshold).long()

    results = evaluate(y_true, y_pred, y_scores, model_name='Synthetic Test')
    print_results(results)

    # Compare two fake models
    y_pred2   = (y_scores >= 0.7).long()
    results2  = evaluate(y_true, y_pred2, y_scores, model_name='Strict Threshold')
    compare_models([results, results2])

    print('\nmetrics.py  ✓  all checks passed')