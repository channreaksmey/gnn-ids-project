# Detailed Description of `utils/metrics.py`

The file `metrics.py` contains the shared evaluation utilities for the project’s intrusion-detection models.

It provides one consistent evaluation process for:

- MAGIC;
- FLASH;
- Random Forest;
- other binary node-classification models.

Using the same functions for every model ensures that their results are calculated with the same formulas and can be compared fairly.

The module evaluates predictions at the **node level**:

```text
0 = benign
1 = malicious
```

It calculates standard classification metrics, confusion-matrix values, ranking-based AUC metrics, ROC and precision-recall curve data, and optimized decision thresholds.

---

# 1. Main responsibilities

The file performs six main tasks:

1. Converts PyTorch tensors or Python lists into NumPy arrays.
2. Calculates classification metrics from true and predicted labels.
3. Calculates ROC-AUC and PR-AUC from continuous anomaly scores.
4. Prints formatted results for one model.
5. Compares multiple models in a single table.
6. Searches for a threshold that maximizes F1, precision, or recall.

The module uses:

```python
torch
numpy
scikit-learn
```

The scikit-learn metrics imported by the file are:

```python
precision_score
recall_score
f1_score
confusion_matrix
roc_auc_score
average_precision_score
roc_curve
precision_recall_curve
```

---

# 2. Input concepts

The evaluation functions use three different kinds of values.

## Ground-truth labels

`y_true` contains the actual labels:

```text
0 = benign
1 = malicious
```

Example:

```python
y_true = torch.tensor([0, 0, 1, 0, 1])
```

## Binary predictions

`y_pred` contains the model’s final decisions:

```text
0 = predicted benign
1 = predicted malicious
```

Example:

```python
y_pred = torch.tensor([0, 1, 1, 0, 0])
```

These values are normally produced after applying a threshold to a score:

```python
y_pred = (y_scores >= threshold).long()
```

## Continuous scores

`y_scores` contains a continuous suspiciousness score for every node.

Higher values should mean that the node is more likely to be malicious.

Example:

```python
y_scores = torch.tensor([
    0.02, 0.81, 0.95, 0.10, 0.30
])
```

Continuous scores are required for:

- ROC-AUC;
- PR-AUC;
- ROC curves;
- precision-recall curves;
- threshold optimization.

A binary prediction alone cannot provide the full ranking information required by these metrics.

---

# 3. `evaluate`

The main function is:

```python
evaluate(
    y_true,
    y_pred,
    y_scores=None,
    model_name='Model'
)
```

Its purpose is to calculate all core metrics for one model.

## Parameters

### `y_true`

The ground-truth labels:

```python
torch.Tensor
```

Expected shape:

```text
(N,)
```

where `N` is the number of evaluated nodes.

### `y_pred`

The binary model predictions:

```python
torch.Tensor
```

Expected values:

```text
0 or 1
```

Expected shape:

```text
(N,)
```

The length of `y_pred` must match the length of `y_true`.

### `y_scores`

Optional continuous anomaly scores:

```python
torch.Tensor
```

Expected shape:

```text
(N,)
```

If this argument is omitted, the function still calculates:

- precision;
- recall;
- F1-score;
- FPR;
- TP;
- FP;
- FN;
- TN.

However, ROC-AUC and PR-AUC remain `None`.

### `model_name`

A label used in printed output:

```python
model_name='FLASH'
```

It is also stored in the returned results dictionary.

---

# 4. Tensor conversion

The function first converts PyTorch tensors into NumPy arrays:

```python
yt = _to_numpy(y_true).astype(int)
yp = _to_numpy(y_pred).astype(int)
```

The helper `_to_numpy` handles both tensors and ordinary Python data:

```python
def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)
```

This is useful because model outputs may be:

- CPU tensors;
- GPU tensors;
- tensors that require gradients;
- NumPy arrays;
- Python lists.

The use of:

```python
detach().cpu().numpy()
```

ensures that metric calculation does not track gradients and works even when the model runs on a GPU.

---

# 5. All-positive and all-negative warnings

The function checks whether the model predicted only one class:

```python
if yp.sum() == 0:
```

If no nodes were predicted malicious, it prints:

```text
WARNING: model predicted NO positives
```

This usually means the detection threshold is too high.

It also checks:

```python
if yp.sum() == len(yp):
```

If every node was predicted malicious, it prints:

```text
WARNING: model predicted ALL positives
```

This often means the threshold is too low.

These warnings do not stop execution. The metrics are still computed.

---

# 6. Precision

Precision answers:

> Of all nodes predicted as malicious, how many were actually malicious?

The formula is:

```text
Precision = TP / (TP + FP)
```

where:

- `TP` = true positives;
- `FP` = false positives.

A high precision means the model generates relatively few false alarms.

For an intrusion-detection system, precision is important because every false positive may require an analyst to investigate a benign node.

The implementation is:

```python
precision = precision_score(
    yt,
    yp,
    zero_division=0
)
```

`zero_division=0` prevents an exception when the model predicts no positive nodes.

---

# 7. Recall

Recall answers:

> Of all truly malicious nodes, how many did the model detect?

The formula is:

```text
Recall = TP / (TP + FN)
```

where:

- `TP` = true positives;
- `FN` = false negatives.

A high recall means the model misses fewer attacks.

Recall is especially important in cybersecurity because a false negative represents malicious activity that was not detected.

The implementation is:

```python
recall = recall_score(
    yt,
    yp,
    zero_division=0
)
```

If the model predicts no malicious nodes, recall becomes zero.

---

# 8. F1-score

F1-score is the harmonic mean of precision and recall:

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

It balances:

- false alarms;
- missed attacks.

The implementation is:

```python
f1 = f1_score(
    yt,
    yp,
    zero_division=0
)
```

F1-score is useful when both precision and recall matter. However, it depends on the selected threshold, unlike ROC-AUC and PR-AUC.

A model can have a different F1-score at every threshold.

---

# 9. Confusion matrix and FPR

The function calculates the confusion matrix using:

```python
tn, fp, fn, tp = confusion_matrix(
    yt,
    yp,
    labels=[0, 1]
).ravel()
```

The four values are:

| Value | Meaning |
|---|---|
| TP | Malicious nodes correctly detected |
| FP | Benign nodes incorrectly flagged |
| FN | Malicious nodes missed |
| TN | Benign nodes correctly ignored |

The false-positive rate is calculated as:

```text
FPR = FP / (FP + TN)
```

It measures the proportion of benign nodes incorrectly classified as malicious.

The implementation protects against division by zero:

```python
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
```

For a security-monitoring system, even a small FPR can create many false alarms when the graph contains hundreds of thousands of benign nodes.

For example, an FPR of `0.001` on 319,448 benign nodes can still produce approximately 319 false positives.

---

# 10. Returned results dictionary

The `evaluate` function returns a dictionary with this structure:

```python
{
    'model': model_name,
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'fpr': fpr,
    'tp': int(tp),
    'fp': int(fp),
    'fn': int(fn),
    'tn': int(tn),
    'roc_auc': None,
    'pr_auc': None,
}
```

If `y_scores` is supplied, `roc_auc` and `pr_auc` are populated.

Example:

```python
results = evaluate(
    y_true=test_labels,
    y_pred=predictions,
    y_scores=scores,
    model_name='FLASH'
)
```

A possible result is:

```python
{
    'model': 'FLASH',
    'precision': 0.42,
    'recall': 0.68,
    'f1': 0.52,
    'fpr': 0.0012,
    'tp': 17203,
    'fp': 237,
    'fn': 8116,
    'tn': 319211,
    'roc_auc': 0.91,
    'pr_auc': 0.74
}
```

---

# 11. ROC-AUC

ROC-AUC measures how well the model ranks malicious nodes above benign nodes over all possible thresholds.

It uses:

```python
roc_auc_score(yt, ys)
```

ROC-AUC is threshold-independent.

Interpretation:

| ROC-AUC | General interpretation |
|---:|---|
| 0.50 | Random ranking |
| 0.60–0.70 | Weak |
| 0.70–0.80 | Fair |
| 0.80–0.90 | Good |
| 0.90–1.00 | Excellent |

ROC-AUC can be useful, but it may appear overly favorable on highly imbalanced datasets. A model may rank many benign nodes correctly while still producing too many false positives for the rare malicious class.

If only one class is present in `y_true`, scikit-learn raises a `ValueError`. The function catches this and leaves:

```python
results['roc_auc'] = None
```

---

# 12. PR-AUC

The function calculates PR-AUC using:

```python
average_precision_score(yt, ys)
```

This is an average-precision summary of the precision-recall curve.

PR-AUC is generally more informative than ROC-AUC for the current dataset because the test graph contains:

```text
7.3% malicious nodes
92.7% benign nodes
```

PR-AUC focuses more directly on the quality of positive-class detection.

A random classifier’s approximate PR-AUC baseline is related to the positive-class prevalence. For this dataset, that baseline is approximately:

```text
0.073
```

Therefore, a PR-AUC substantially above `0.073` indicates useful ranking performance, while a value near the malicious prevalence indicates weak performance.

The code correctly emphasizes PR-AUC:

```python
# Our data: 7.3% malicious — PR-AUC is the key metric
```

---

# 13. `print_results`

The function:

```python
print_results(results)
```

prints one formatted report for a model.

Example usage:

```python
print_results(results)
```

The output format is:

```text
==================================================
  FLASH
==================================================
  Precision  : 0.4200
  Recall     : 0.6800
  F1-Score   : 0.5200
  FPR        : 0.0012
  ROC-AUC    : 0.9100
  PR-AUC     : 0.7400
  ── Confusion matrix ──
  TP=17,203  FP=237  FN=8,116  TN=319,211
==================================================
```

The function always prints:

- precision;
- recall;
- F1-score;
- FPR;
- TP;
- FP;
- FN;
- TN.

It prints ROC-AUC and PR-AUC only when those values are available.

---

# 14. `compare_models`

The function:

```python
compare_models(results_list)
```

prints a side-by-side comparison of multiple model result dictionaries.

Example:

```python
flash_results = evaluate(
    y_true,
    flash_predictions,
    flash_scores,
    model_name='FLASH'
)

magic_results = evaluate(
    y_true,
    magic_predictions,
    magic_scores,
    model_name='MAGIC'
)

rf_results = evaluate(
    y_true,
    rf_predictions,
    rf_scores,
    model_name='Random Forest'
)

compare_models([
    flash_results,
    magic_results,
    rf_results
])
```

The comparison table includes:

- model name;
- precision;
- recall;
- F1;
- FPR;
- PR-AUC.

Example format:

```text
===========================================================================
  MODEL COMPARISON
===========================================================================
  Model                    Precision     Recall        F1        FPR     PR-AUC
  ----------------------  -----------  ---------  ---------  ---------  ---------
  FLASH                       0.4200     0.6800     0.5200     0.0012     0.7400
  MAGIC                       0.3900     0.7100     0.5000     0.0015     0.7100
  Random Forest               0.5100     0.6200     0.5600     0.0009     0.7600
===========================================================================
  PR-AUC is the primary metric (data is 7.3% malicious / imbalanced)
===========================================================================
```

The function intentionally prioritizes PR-AUC because accuracy and ROC-AUC can be misleading when benign nodes dominate the dataset.

---

# 15. `get_roc_curve_data`

The function:

```python
get_roc_curve_data(y_true, y_scores)
```

returns the arrays required to plot an ROC curve:

```python
fpr_arr, tpr_arr, thresholds
```

Internally, it calls:

```python
roc_curve(yt, ys)
```

The returned arrays mean:

- `fpr_arr`: false-positive-rate values;
- `tpr_arr`: true-positive-rate values, equivalent to recall;
- `thresholds`: score thresholds used to create the curve.

Example:

```python
from utils.metrics import get_roc_curve_data
import matplotlib.pyplot as plt

fpr, tpr, thresholds = get_roc_curve_data(
    test_labels,
    scores
)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
```

A random classifier is represented by the diagonal line:

```python
plt.plot([0, 1], [0, 1], '--')
```

---

# 16. `get_pr_curve_data`

The function:

```python
get_pr_curve_data(y_true, y_scores)
```

returns:

```python
precision_array, recall_array, thresholds
```

Internally, it calls:

```python
precision_recall_curve(yt, ys)
```

Example:

```python
from utils.metrics import get_pr_curve_data
import matplotlib.pyplot as plt

precision, recall, thresholds = get_pr_curve_data(
    test_labels,
    scores
)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()
```

For this dataset, the precision-recall curve is particularly important because the positive class is relatively rare.

A useful reference line is the malicious prevalence:

```python
positive_rate = test_labels.float().mean().item()

plt.axhline(
    positive_rate,
    linestyle='--',
    label=f'Baseline: {positive_rate:.3f}'
)
```

---

# 17. `find_best_threshold`

The function:

```python
find_best_threshold(
    y_true,
    y_scores,
    metric='f1',
    n_steps=100
)
```

searches for the score threshold that maximizes a selected metric.

It supports:

```text
'f1'
'precision'
'recall'
```

The function creates evenly spaced thresholds:

```python
thresholds = np.linspace(
    ys.min(),
    ys.max(),
    n_steps
)
```

For each threshold, it creates binary predictions:

```python
yp = (ys >= t).astype(int)
```

It then calculates the requested metric.

The function returns:

```python
best_threshold
best_score
all_results
```

where `all_results` is a list of:

```python
(threshold, metric_value)
```

Example:

```python
threshold, best_f1, history = find_best_threshold(
    test_labels,
    scores,
    metric='f1',
    n_steps=200
)

predictions = (scores >= threshold).long()

results = evaluate(
    test_labels,
    predictions,
    scores,
    model_name='FLASH'
)
```

---

# 18. Choosing the threshold metric

## Maximize F1

Use:

```python
metric='f1'
```

This balances precision and recall.

This is a reasonable default when the project wants an overall balance between missed attacks and false alarms.

## Maximize precision

Use:

```python
metric='precision'
```

This reduces false alarms, but it may produce poor recall or no positive predictions.

This is appropriate when analyst investigation capacity is limited.

## Maximize recall

Use:

```python
metric='recall'
```

This prioritizes detecting as many malicious nodes as possible.

However, it can result in many false positives.

This is appropriate when missing an attack is much more costly than investigating benign nodes.

---

# 19. Synthetic test mode

The file can be run directly:

```bash
python utils/metrics.py
```

The `__main__` section performs a synthetic evaluation.

It generates:

```python
N = 10000
y_true = (torch.rand(N) < 0.073).long()
```

This simulates a dataset with approximately 7.3% malicious nodes, matching the project’s test-set class ratio.

It then creates random scores and shifts malicious-node scores upward:

```python
y_scores[y_true == 1] += 0.3
```

This creates an artificial model with some ability to distinguish malicious nodes.

The test then:

1. finds the best F1 threshold;
2. creates binary predictions;
3. evaluates the synthetic model;
4. prints its results;
5. evaluates a second model with a fixed threshold of `0.7`;
6. compares both models.

Run it with:

```bash
python utils/metrics.py
```

A successful run ends with:

```text
metrics.py  ✓  all checks passed
```

This is a smoke test for the metrics implementation, not a real model evaluation.

---

# 20. Typical use with `data_loader.py`

The metrics module can be combined with `data_loader.py` like this:

```python
from utils.data_loader import load_dataset
from utils.metrics import evaluate, print_results

train_graphs, test_graph, test_labels = load_dataset('./data')

# Example model output
scores = model_scores

# Convert scores to binary predictions
threshold = 0.5
predictions = (scores >= threshold).long()

results = evaluate(
    y_true=test_labels,
    y_pred=predictions,
    y_scores=scores,
    model_name='FLASH'
)

print_results(results)
```

To optimize the threshold:

```python
from utils.metrics import find_best_threshold

threshold, score, history = find_best_threshold(
    test_labels,
    scores,
    metric='f1'
)

predictions = (scores >= threshold).long()
```

---

# 21. Important score-direction requirement

The metrics assume:

```text
higher y_scores = more suspicious = more likely malicious
```

This must be true for every model.

For example:

- Random Forest class-1 probability normally satisfies this;
- an anomaly score normally satisfies this;
- a KNN distance normally satisfies this;
- a benign-class confidence does not satisfy this unless it is inverted.

The file documentation describes model-specific score conventions:

```text
MAGIC: KNN distance score
FLASH: 1 - classifier confidence
RF: classifier probability for class 1
```

If a model instead produces a benign probability, convert it:

```python
malicious_scores = 1.0 - benign_probability
```

If score direction is reversed, ROC-AUC and PR-AUC can become misleading or worse than random.

---

# 22. Important limitations and possible fixes

## A. `evaluate` does not validate input shapes

Currently, the function assumes that:

```python
len(y_true) == len(y_pred)
```

and, when supplied:

```python
len(y_true) == len(y_scores)
```

A safer implementation should explicitly validate this before calling scikit-learn.

It should also validate that:

- labels contain only 0 and 1;
- predictions contain only 0 and 1;
- scores are one-dimensional;
- score values are finite.

Without those checks, shape errors or invalid values may produce confusing exceptions.

---

## B. ROC-AUC and PR-AUC errors are silently ignored

The code uses:

```python
except ValueError:
    pass
```

This allows evaluation to continue when only one class is present. However, it hides the reason that an AUC metric could not be calculated.

A better approach would be to set the metric to `None` and optionally print a warning such as:

```text
WARNING: ROC-AUC unavailable because y_true contains only one class.
```

This makes the result easier to interpret.

---

## C. Threshold search can overfit the test set

The function can optimize a threshold using:

```python
test_labels
test_scores
```

If that threshold is then reported as the final test result, the test set has influenced model selection. This can make the evaluation optimistic.

A better workflow is:

1. train the model on training data;
2. generate validation scores;
3. select the threshold on a validation set;
4. freeze the threshold;
5. evaluate once on `test0.pkl`.

For the current project, this is especially important because the test graph contains the malicious labels.

---

## D. Threshold search uses an evenly spaced grid

The current approach uses:

```python
np.linspace(ys.min(), ys.max(), n_steps)
```

This may miss the exact threshold that gives the best result, especially when scores are clustered.

A more precise approach is to evaluate thresholds derived from the unique score values or from the thresholds returned by `precision_recall_curve`.

The current function is simple and usually adequate for a first experiment, but it is approximate.

---

## E. Constant or empty scores need protection

If `y_scores` is empty or contains only one value, threshold search may not be meaningful.

Examples of potential problems:

```python
ys.min()
ys.max()
```

can fail for an empty array, and a constant score cannot meaningfully rank nodes.

The function should validate that:

```python
len(y_scores) > 0
np.isfinite(y_scores).all()
```

and should handle:

```python
ys.min() == ys.max()
```

explicitly.

---

## F. Precision maximization may be misleading

If `metric='precision'`, a threshold that flags only one highly confident node may achieve precision of 1.0 while detecting almost none of the malicious nodes.

Therefore, precision should normally be evaluated together with:

- recall;
- F1-score;
- number of predicted positives;
- false-positive count.

A minimum recall or minimum number of detections may be required in a production setting.

---

## G. PR-AUC naming

The implementation uses:

```python
average_precision_score
```

and stores it as:

```python
pr_auc
```

Strictly speaking, average precision is not always identical to trapezoidal area under a precision-recall curve. In many machine-learning reports, it is still described as PR-AUC, but the project should document that it uses scikit-learn’s **average precision** calculation.

---

## H. FPR alone may not communicate operational impact

With hundreds of thousands of benign nodes, a small FPR can still generate many false alarms.

It is useful to report both:

```text
FPR
FP count
```

The module already reports both, which is good. Model comparisons should not compare FPR without also looking at the absolute false-positive count.

---

# 23. Recommended evaluation workflow

A reliable evaluation workflow is:

```python
from utils.metrics import (
    evaluate,
    print_results,
    compare_models,
    find_best_threshold,
)

# 1. Generate continuous scores on validation data.
validation_scores = ...

# 2. Select a threshold using validation labels.
threshold, _, _ = find_best_threshold(
    validation_labels,
    validation_scores,
    metric='f1'
)

# 3. Generate scores on the test graph.
test_scores = ...

# 4. Convert test scores to binary predictions.
test_predictions = (
    test_scores >= threshold
).long()

# 5. Evaluate the frozen threshold.
test_results = evaluate(
    y_true=test_labels,
    y_pred=test_predictions,
    y_scores=test_scores,
    model_name='FLASH'
)

# 6. Print the final result.
print_results(test_results)
```

For multiple models:

```python
all_results = [
    flash_results,
    magic_results,
    rf_results,
]

compare_models(all_results)
```

The primary model-selection metric should be PR-AUC or a validation-set F1-score because the data is imbalanced.

---

# 24. Summary

`metrics.py` is the project’s shared evaluation layer. It makes model results comparable by applying the same formulas to every model.

Its most important functions are:

| Function | Purpose |
|---|---|
| `evaluate` | Calculate classification and AUC metrics |
| `print_results` | Display one model’s metrics |
| `compare_models` | Compare multiple models |
| `get_roc_curve_data` | Prepare ROC plotting data |
| `get_pr_curve_data` | Prepare precision-recall plotting data |
| `find_best_threshold` | Select a threshold based on F1, precision, or recall |
| `_to_numpy` | Convert tensors or lists to NumPy arrays |

For this project, the most important metrics are:

1. **PR-AUC**, because the malicious class is only 7.3%;
2. **Recall**, because missed attacks are false negatives;
3. **Precision**, because false alarms consume analyst time;
4. **F1-score**, because it balances precision and recall;
5. **FPR and FP count**, because benign nodes are numerous;
6. **ROC-AUC**, as a general ranking metric but not the only measure of success.

The file is currently suitable for shared model evaluation and experimentation. The most valuable future improvements would be stronger input validation, clearer warnings for unavailable AUC metrics, validation-based threshold selection, and a precise resolution of the project’s label and score definitions.