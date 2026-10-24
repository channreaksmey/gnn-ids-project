# Detailed Description of `models/baseline_rf.py`

The file `baseline_rf.py` implements a **Random Forest anomaly-detection baseline** for the provenance-graph intrusion-detection project.

It is a non-GNN model used to compare against graph-based models such as FLASH and MAGIC. Instead of performing message passing over the graph, it converts each node into a fixed-length feature vector containing node attributes and handcrafted local graph-structure information.

The model is trained using the benign training graphs and produces an anomaly score for every node in the test graph.

---

## 1. Purpose of the model

The purpose of this file is to provide a simple, interpretable baseline:

```text
DGL graph
    ↓
Handcrafted node features
    ↓
Random Forest node-type classifier
    ↓
Node-type probability distribution
    ↓
Entropy-based anomaly score
    ↓
Malicious/benign prediction
```

The model does not directly train on malicious labels. Instead, it learns the distribution of benign node types from the training graphs.

At test time, nodes that produce uncertain node-type predictions are considered anomalous.

The basic assumption is:

> A benign node should resemble one of the node patterns observed during training. A node that does not fit any learned pattern will produce a less confident prediction and receive a higher anomaly score.

---

# 2. Main class

The main class is:

```python
RandomForestBaseline
```

It contains methods for:

- initializing the Random Forest;
- extracting node features;
- building edge-type histograms;
- fitting the model;
- generating anomaly scores;
- generating binary predictions;
- saving the fitted model;
- loading a saved model.

Example import:

```python
from models.baseline_rf import RandomForestBaseline
```

Example initialization:

```python
model = RandomForestBaseline()
```

---

# 3. Model configuration

The constructor is:

```python
RandomForestBaseline(
    n_estimators=200,
    max_depth=8,
    n_jobs=-1,
    random_state=42
)
```

The underlying classifier is:

```python
RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=50,
    n_jobs=n_jobs,
    random_state=random_state,
    class_weight='balanced'
)
```

## Parameters

### `n_estimators`

Number of decision trees in the forest.

Default:

```text
200
```

Increasing this generally makes the model more stable, but also increases:

- training time;
- memory usage;
- model size.

### `max_depth`

Maximum depth of each decision tree.

Default:

```text
8
```

The shallow depth limits model complexity and reduces overfitting.

Earlier versions used a deeper forest. The current setting is intended to avoid highly specific trees that memorize the training data and produce overconfident probability estimates.

### `min_samples_leaf`

Each leaf must contain at least 50 training samples:

```python
min_samples_leaf=50
```

This regularizes the Random Forest by preventing leaves that represent only a few individual nodes.

A larger value generally produces smoother, less extreme probabilities. A smaller value may capture finer patterns but can overfit.

### `n_jobs`

Number of CPU workers.

Default:

```text
-1
```

This uses all available CPU cores.

### `random_state`

Random seed used for reproducibility.

Default:

```text
42
```

### `class_weight='balanced'`

This compensates for imbalance among the node-type classes used during training.

The training target is node type, not malicious/benign status. Therefore, this setting balances node-type frequencies rather than directly balancing the attack classes.

---

# 4. Constants

The file defines:

```python
NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 10
```

These values must match the dataset and `data_loader.py`.

The supported categories are:

```text
Node types: 0, 1, 2, 3, 4
Edge types: 0, 1, 2, ..., 9
```

These constants determine the dimensions of the one-hot node-type features and edge-type histograms.

---

# 5. Input graph format

The class expects a DGL graph containing:

```python
graph.ndata['attr']
graph.ndata['type']
graph.edata['type']
```

The expected fields are:

| Field | Shape | Meaning |
|---|---|---|
| `ndata["attr"]` | `(N, 5)` | Raw node attributes |
| `ndata["type"]` | `(N,)` | Integer node type |
| `edata["type"]` | `(E,)` | Integer edge type |

The model does not directly use `graph.edata["attr"]`, the 17-dimensional edge-feature tensor. Instead, it uses the categorical edge types to build incoming-edge histograms.

---

# 6. Feature engineering

Each node is represented by 23 features.

## Feature summary

| Feature group | Number of features |
|---|---:|
| Raw node attributes | 5 |
| One-hot node type | 5 |
| In-degree | 1 |
| Out-degree | 1 |
| Total degree | 1 |
| Incoming edge-type histogram | 10 |
| **Total** | **23** |

The final feature matrix has shape:

```text
(num_nodes, 23)
```

For example:

```text
train0: (167265, 23)
```

Each row represents one node.

---

## 6.1 Raw node attributes

The first five features come from:

```python
graph.ndata['attr']
```

The code converts the tensor to NumPy:

```python
node_attr = graph.ndata['attr'].numpy()
```

The expected shape is:

```text
(num_nodes, 5)
```

These are the original node-level attributes from the dataset.

The current implementation assumes that the tensor is stored on the CPU. If the graph data is moved to a GPU, this conversion should be changed to:

```python
graph.ndata['attr'].detach().cpu().numpy()
```

---

## 6.2 One-hot node type

Each node has a categorical type from 0 through 4.

The code converts this categorical value into a five-dimensional one-hot vector:

```python
type_onehot = np.zeros(
    (num_nodes, NUM_NODE_TYPES)
)

type_onehot[
    np.arange(num_nodes),
    node_type
] = 1.0
```

Examples:

```text
Type 0 → [1, 0, 0, 0, 0]
Type 1 → [0, 1, 0, 0, 0]
Type 2 → [0, 0, 1, 0, 0]
Type 3 → [0, 0, 0, 1, 0]
Type 4 → [0, 0, 0, 0, 1]
```

This contributes five features.

### Important consequence

The one-hot node type is included in the input features, and the same node type is used as the Random Forest target during training.

Therefore, the model is given direct information about the category it is trained to predict. This explains why the training node-type accuracy is expected to be close to 1.0.

---

## 6.3 Degree features

The model calculates:

```python
in_deg = graph.in_degrees()
out_deg = graph.out_degrees()
tot_deg = in_deg + out_deg
```

These produce three features per node:

1. **In-degree**: number of edges pointing into the node;
2. **Out-degree**: number of edges leaving the node;
3. **Total degree**: sum of in-degree and out-degree.

The degree arrays are reshaped into column vectors:

```python
in_deg.reshape(-1, 1)
out_deg.reshape(-1, 1)
```

This gives each node a basic description of its local activity level.

---

## 6.4 Incoming edge-type histogram

The method:

```python
_build_edge_type_histogram(graph)
```

creates ten features for every node.

For each node, it counts how many incoming edges belong to each edge type.

The raw histogram has the form:

```text
[
    count_type_0,
    count_type_1,
    ...
    count_type_9
]
```

For example:

```text
[4, 2, 0, 1, 0, 0, 0, 0, 3, 0]
```

The histogram is normalized by the number of incoming edges:

```python
row_sums = hist.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
hist = hist / row_sums
```

The result represents proportions rather than absolute counts.

Example:

```text
[0.4, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0]
```

This provides information about the kinds of interactions associated with each node without using full GNN message passing.

### Important limitation

Only incoming edge types are included. Outgoing edge-type distributions are not currently represented.

The model also discards the absolute number of incoming edges when it normalizes the histogram. Two nodes with the same proportions but very different numbers of edges can receive the same histogram features.

---

# 7. `extract_features`

The public feature-extraction method is:

```python
extract_features(graph)
```

It returns:

```python
X, node_types
```

where:

- `X` is the 23-dimensional feature matrix;
- `node_types` is the node-type vector.

Signature:

```python
def extract_features(
    graph: dgl.DGLGraph
) -> Tuple[np.ndarray, np.ndarray]:
```

Expected return values:

```text
X:
    shape (num_nodes, 23)

node_types:
    shape (num_nodes,)
```

Example:

```python
X, node_types = model.extract_features(train_graphs[0])

print(X.shape)
# (167265, 23)

print(node_types.shape)
# (167265,)
```

---

# 8. `fit`

The training method is:

```python
fit(train_graphs, verbose=True)
```

It accepts a list of DGL graphs.

Example:

```python
model.fit(train_graphs)
```

## Training procedure

For each graph, the method:

1. extracts the 23-dimensional features;
2. extracts the graph’s node types;
3. stores the feature matrix;
4. stores the node-type targets.

The features from all training graphs are combined:

```python
X_train = np.vstack(all_X)
```

The node-type targets are combined:

```python
y_train = np.concatenate(all_y)
```

For the current dataset, the total training size is:

```text
1,279,199 nodes
2,246,714 edges
23 features per node
```

The node-type distribution is:

| Node type | Number of nodes | Percentage |
|---:|---:|---:|
| 0 | 47,229 | 3.7% |
| 1 | 144,815 | 11.3% |
| 2 | 1,028,338 | 80.4% |
| 3 | 58,806 | 4.6% |
| 4 | 11 | approximately 0.0% |

---

# 9. Training target

The model is trained using:

```python
all_y.append(node_types)
```

Therefore, the Random Forest target is:

```text
node type 0, 1, 2, 3, or 4
```

It is not trained directly on:

```text
benign = 0
malicious = 1
```

The malicious labels returned by `load_dataset()` are not used during model fitting.

The approach is therefore an anomaly-detection strategy:

1. learn benign node-type patterns;
2. classify test nodes by node type;
3. use uncertainty as the anomaly score.

This is appropriate for a setting where the training graphs have no attack labels.

---

# 10. Feature scaling

Before training, the model applies:

```python
self.scaler.fit_transform(X_train)
```

The scaler is a `StandardScaler`.

During inference, the same fitted scaler is applied:

```python
self.scaler.transform(X_test)
```

Using the same scaler for training and testing ensures that both feature matrices use the same transformation.

Although scaling is not generally necessary for Random Forests, it is harmless in this case and keeps the preprocessing pipeline explicit.

Tree-based models are usually insensitive to monotonic scaling. The scaler would be more important for models such as:

- logistic regression;
- SVM;
- KNN;
- neural networks.

---

# 11. Training accuracy

After fitting, the method predicts the training samples:

```python
train_preds = self.rf.predict(X_train_scaled)
train_acc = (train_preds == y_train).mean()
```

It prints:

```text
Training node-type accuracy : 1.0000
(expected ~1.0 — type is an input feature)
```

This message is intentionally explicit.

The perfect accuracy is expected because:

1. node type is part of the input;
2. node type is the training target;
3. the Random Forest can directly learn the relationship.

This is not a malicious-detection accuracy score.

It does not prove that:

- the model detects every malicious node;
- the model generalizes to the test graph;
- the anomaly score is perfect;
- the test F1-score will be 1.0.

The relevant detection metrics are calculated later using the test malicious labels.

---

# 12. `predict`

The inference method is:

```python
predict(test_graph, threshold=None)
```

It returns:

```python
scores, preds
```

where:

- `scores` is a continuous anomaly score for every node;
- `preds` is a binary prediction vector.

Signature:

```python
def predict(
    test_graph: dgl.DGLGraph,
    threshold: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
```

---

## 12.1 Fitted-model check

Before inference, the method verifies:

```python
if not self.is_fitted:
    raise RuntimeError('Call fit() before predict().')
```

This prevents inference with an untrained model.

---

## 12.2 Test feature extraction

The test graph is converted into the same 23-dimensional representation:

```python
X_test, _ = self.extract_features(test_graph)
X_test_scaled = self.scaler.transform(X_test)
```

The test matrix has shape:

```text
(344,767, 23)
```

---

## 12.3 Node-type probabilities

The Random Forest produces a probability distribution over node types:

```python
proba = self.rf.predict_proba(X_test_scaled)
```

The expected shape is:

```text
(num_test_nodes, number_of_node_types)
```

For this dataset, the probability columns correspond to node types 0 through 4.

For each node, the probabilities should sum to approximately 1:

```text
P(type 0) + P(type 1) + ... + P(type 4) ≈ 1
```

---

# 13. Entropy-based anomaly score

The current model uses normalized entropy rather than simply calculating:

```python
1 - max_probability
```

The probability values are clipped:

```python
proba_clipped = np.clip(
    proba,
    1e-10,
    1.0
)
```

This prevents:

```python
log(0)
```

from causing numerical problems.

The entropy is calculated as:

```python
entropy = -np.sum(
    proba_clipped * np.log(proba_clipped),
    axis=1
)
```

The maximum entropy for five classes is:

```python
np.log(5)
```

The score is normalized:

```python
scores = entropy / max_entropy
```

The result is approximately within:

```text
0 = confident prediction
1 = maximally uncertain prediction
```

## Interpretation

A low score means the Random Forest strongly favors one node type:

```text
[0.99, 0.01, 0.00, 0.00, 0.00]
```

A high score means the probabilities are distributed more evenly:

```text
[0.20, 0.20, 0.20, 0.20, 0.20]
```

The model treats high uncertainty as evidence that the node does not fit a known benign pattern.

## Important qualification

Entropy is an anomaly score, not a calibrated probability of maliciousness.

A malicious node may receive a low entropy score if it resembles normal behavior. A benign node may receive a high entropy score if it has an unusual structure.

The test results indicate that entropy is highly correlated with the malicious labels for this dataset, but the score should still be interpreted as anomaly likelihood rather than direct attack probability.

---

# 14. Thresholding

If `threshold=None`, the method returns all-zero binary predictions:

```python
preds = torch.zeros(
    len(scores),
    dtype=torch.long
)
```

This allows the caller to retrieve scores without committing to a threshold.

If a threshold is provided:

```python
preds = (
    scores_tensor >= threshold
).long()
```

The rule is:

```text
score >= threshold → malicious
score < threshold  → benign
```

A lower threshold creates more alerts.

A higher threshold creates fewer alerts.

---

# 15. Command-line execution

The file can be executed directly:

```bash
python models/baseline_rf.py ./data
```

If no directory is supplied:

```bash
python models/baseline_rf.py
```

the default path is:

```text
./data
```

The script adds the project root to `sys.path`:

```python
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
```

This allows it to import:

```python
from utils.data_loader import load_dataset
from utils.metrics import evaluate
```

---

# 16. Command-line workflow

When executed directly, the file performs the following steps.

## Step 1: Load the dataset

```python
train_graphs, test_graph, test_labels = load_dataset(data_dir)
```

The four training graphs are loaded, followed by `test0.pkl`.

The test labels are built from the malicious node IDs in `metadata.json`.

---

## Step 2: Create the model

```python
model = RandomForestBaseline(
    n_estimators=200,
    max_depth=8
)
```

The model uses:

```text
200 trees
maximum depth 8
minimum leaf size 50
balanced node-type weights
```

---

## Step 3: Train on benign graphs

```python
model.fit(
    train_graphs,
    verbose=True
)
```

The model learns to classify node types using the benign training graphs.

---

## Step 4: Generate test scores

```python
scores, _ = model.predict(
    test_graph,
    threshold=None
)
```

This produces entropy scores without applying a binary threshold.

---

## Step 5: Find a threshold

```python
best_thresh, best_f1, _ = find_best_threshold(
    test_labels,
    scores,
    metric='f1'
)
```

The threshold is selected to maximize F1-score.

Because the dataset has no separate labeled validation graph, the current project calibrates the threshold using the labeled test graph. This follows the evaluation setting used for comparison with the original MAGIC experiment.

---

## Step 6: Generate final predictions

```python
_, preds = model.predict(
    test_graph,
    threshold=best_thresh
)
```

Each node is classified as:

```text
0 = benign
1 = malicious
```

---

## Step 7: Evaluate

```python
results = evaluate(
    test_labels,
    preds,
    scores,
    model_name='Random Forest Baseline'
)
```

The shared metrics module calculates:

- precision;
- recall;
- F1-score;
- FPR;
- ROC-AUC;
- PR-AUC;
- TP;
- FP;
- FN;
- TN.

---

# 17. Current experimental result

The latest command output is:

```text
Best threshold : 0.3196  →  F1 = 0.9509

Precision  : 0.9350
Recall     : 0.9673
F1-Score   : 0.9509
FPR        : 0.0053
ROC-AUC    : 0.9708
PR-AUC     : 0.8807

TP=24,492
FP=1,702
FN=827
TN=317,746
```

## Precision

```text
0.9350
```

Approximately 93.5% of flagged nodes are malicious.

## Recall

```text
0.9673
```

Approximately 96.7% of malicious nodes are detected.

## F1-score

```text
0.9509
```

This indicates a strong balance between precision and recall.

## False-positive rate

```text
0.0053
```

Approximately 0.53% of benign nodes are incorrectly flagged.

Because the test graph contains many benign nodes, this still produces:

```text
1,702 false positives
```

## ROC-AUC

```text
0.9708
```

This indicates that the anomaly scores provide strong ranking separation between benign and malicious nodes.

## PR-AUC

```text
0.8807
```

This is especially important for the imbalanced test graph, where only 7.3% of nodes are malicious.

The approximate random PR-AUC baseline is:

```text
0.0734
```

Therefore, `0.8807` represents a substantial improvement over random ranking.

---

# 18. Threshold-evaluation protocol

The script selects the threshold using:

```python
test_labels
```

This is necessary in the current dataset setting because:

- the training graphs have no attack labels;
- there is no separate validation graph;
- only the test graph contains malicious labels;
- the original MAGIC experiment faces the same constraint.

This should be documented clearly.

A suitable report statement is:

> Since the dataset provides no labeled validation split, the anomaly threshold was calibrated using the labeled test graph, following the evaluation protocol used by the original MAGIC study. ROC-AUC and PR-AUC measure threshold-independent ranking performance, while precision, recall, F1-score, FPR, and the confusion matrix represent performance at the test-calibrated threshold.

This distinguishes the ranking metrics from the threshold-dependent operating-point metrics.

---

# 19. Threshold-independent versus threshold-dependent metrics

## Threshold-independent metrics

These evaluate the full score ranking:

- ROC-AUC;
- PR-AUC.

The current results are:

```text
ROC-AUC = 0.9708
PR-AUC  = 0.8807
```

These metrics do not depend on the selected threshold of `0.3196`.

## Threshold-dependent metrics

These depend on the selected threshold:

- precision;
- recall;
- F1-score;
- FPR;
- TP;
- FP;
- FN;
- TN.

These describe the model’s performance at the selected operating point.

---

# 20. Model persistence

The class provides:

```python
save(path)
load(path)
```

## Saving a model

```python
model.save('./out/random_forest_baseline.pkl')
```

The saved file contains:

```python
{
    'rf': self.rf,
    'scaler': self.scaler,
    'is_fitted': self.is_fitted,
    'feature_dim': self.feature_dim
}
```

The model and scaler are saved together so that test-time preprocessing is identical to training-time preprocessing.

## Loading a model

```python
model = RandomForestBaseline()
model.load('./out/random_forest_baseline.pkl')
```

After loading:

```python
scores, predictions = model.predict(
    test_graph,
    threshold=0.3196
)
```

The threshold is not saved as part of the model state. If the threshold is needed later, it should be stored separately in a configuration file or evaluation report.

---

# 21. Important limitations

## 21.1 The model is not directly supervised for attacks

The Random Forest learns node types, not malicious labels.

Therefore, the model is an anomaly detector rather than a conventional supervised attack classifier.

---

## 21.2 Node type is both input and target

The one-hot node type is included among the input features and is also the target during training.

This makes node-type training accuracy uninformative for anomaly-detection quality.

The explicit output message correctly documents this:

```text
expected ~1.0 — type is an input feature
```

---

## 21.3 Entropy is not a calibrated probability

Anomaly scores should not be interpreted as:

```text
score = probability of attack
```

They indicate uncertainty in the node-type prediction.

---

## 21.4 Edge attributes are not used

The graph contains 17-dimensional edge attributes:

```python
graph.edata['attr']
```

but this model does not use them.

It uses only:

```python
graph.edata['type']
```

to build an incoming edge-type histogram.

---

## 21.5 Only incoming edge types are included

Outgoing edge-type patterns are not currently used.

Adding outgoing histograms could provide additional information about node behavior.

---

## 21.6 Absolute edge counts are partly discarded

The edge-type histogram is normalized by the total number of incoming edges.

This captures interaction proportions but loses some activity-volume information.

A node with one incoming edge and a node with 1,000 incoming edges may have identical normalized histograms.

---

## 21.7 Threshold selection uses the labeled test graph

This is necessary for the current paper-compatible evaluation protocol, but it means threshold-dependent results represent a test-calibrated operating point.

They should not be described as a completely independent deployment estimate.

---

## 21.8 Training accuracy is not test performance

The `1.0000` training node-type accuracy only indicates that the Random Forest learned the node-type feature relationship.

The test metrics are the meaningful indicators of anomaly-detection performance.

---

# 22. Recommended future improvements

Potential improvements include:

1. Add explicit validation of input tensor shapes and types.
2. Use CPU-safe conversions such as `.detach().cpu().numpy()`.
3. Add outgoing edge-type histograms.
4. Preserve raw edge counts in addition to normalized proportions.
5. Incorporate edge attributes from `edata["attr"]`.
6. Add neighborhood statistics, such as neighbor-type counts.
7. Save the selected threshold with the model.
8. Report score distributions for benign and malicious nodes.
9. Compare entropy against other uncertainty measures.
10. Run repeated experiments with multiple random seeds.
11. Compare the Random Forest with a direct supervised classifier if labeled training attacks become available.
12. Use paper-compatible test calibration consistently across MAGIC, FLASH, and Random Forest.

---

# 23. Summary

`baseline_rf.py` implements a non-GNN Random Forest anomaly detector for provenance graphs.

It uses 23 handcrafted features per node:

```text
5 raw node attributes
5 one-hot node-type features
3 degree features
10 incoming edge-type proportions
```

The model is trained on benign graphs to classify node types. During test inference, it converts the Random Forest’s node-type probability distribution into a normalized entropy score:

```text
low entropy  → confident normal-pattern prediction
high entropy → uncertain or anomalous prediction
```

The current experiment achieves:

```text
Precision: 0.9350
Recall:    0.9673
F1-score:  0.9509
FPR:       0.0053
ROC-AUC:   0.9708
PR-AUC:    0.8807
```

These results indicate strong anomaly-ranking performance and a strong test-calibrated operating point. The model is a useful baseline for determining whether the more complex FLASH and MAGIC graph-neural-network approaches provide additional value beyond handcrafted local graph features.