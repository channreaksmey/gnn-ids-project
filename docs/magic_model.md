# MAGIC Model Output Explanation

The command:

```bash
python models/magic_model.py ./data
```

successfully trained and evaluated the MAGIC model implemented in `magic_model.py`.

MAGIC stands for **Masked Graph Representation Learning for APT Detection**. It learns node embeddings from benign provenance graphs and detects suspicious test nodes by measuring their distance from benign training embeddings.

---

## 1. MAGIC pipeline

The current implementation follows this process:

```text
Benign training graphs
    ↓
Combine node attributes and node types
    ↓
Randomly mask 50% of node features
    ↓
Encode graphs with GAT layers
    ↓
Reconstruct masked node features
    ↓
Train using scaled cosine reconstruction loss
    ↓
Generate embeddings for all benign training nodes
    ↓
Build a BallTree KNN index
    ↓
Encode test nodes
    ↓
Measure distance to nearest benign embeddings
    ↓
Use distance as anomaly score
    ↓
Choose threshold and classify nodes
```

The model does not use malicious labels during training. The test labels are used only to evaluate the anomaly scores and select the operating threshold.

---

## 2. Dataset loaded

The model uses the same dataset pipeline as the Random Forest baseline.

Training graphs:

```text
train0.pkl: 167,265 nodes, 284,562 edges
train1.pkl: 380,204 nodes, 674,723 edges
train2.pkl: 342,624 nodes, 587,111 edges
train3.pkl: 389,106 nodes, 700,318 edges
```

Total training data:

```text
1,279,199 nodes
2,246,714 edges
```

Test graph:

```text
test0.pkl: 344,767 nodes, 628,107 edges
```

Test labels:

```text
319,448 benign nodes
25,319 malicious nodes
7.3% malicious
```

The training graphs are benign-only. The malicious labels are available only for the test graph through `metadata.json`.

---

# 3. MAGIC architecture

The implementation contains three main components:

1. `GATEncoder`
2. `GATDecoder`
3. `MAGICModel`

---

## 3.1 GAT encoder

The `GATEncoder` uses Graph Attention Network layers from DGL:

```python
from dgl.nn import GATConv
```

The encoder aggregates information from neighboring nodes using learned attention weights.

The current configuration is:

```text
Input dimension: 10
Hidden dimension: 64
Number of GAT layers: 3
Number of attention heads: 1
Dropout: 0.1
```

The input dimension is 10 because the model uses combined node features:

```text
5 raw node attributes
5 one-hot node-type features
```

The encoder concatenates:

- the original 10-dimensional input;
- the output of GAT layer 1;
- the output of GAT layer 2;
- the output of GAT layer 3.

Therefore, the final embedding dimension is:

```text
10 + 64 + 64 + 64 = 202
```

This matches the output:

```text
Embedding dimension: 202
```

The embedding contains information from increasingly wide neighborhoods:

- original node features;
- approximately one-hop information;
- approximately two-hop information;
- approximately three-hop information.

---

## 3.2 GAT decoder

The decoder is another GAT layer.

Its purpose is to reconstruct the original node features from the learned embeddings.

During training:

```text
masked input features
    ↓
GAT encoder
    ↓
node embeddings
    ↓
re-masking projection
    ↓
GAT decoder
    ↓
reconstructed node features
```

The decoder produces reconstructed vectors with the same dimension as the input:

```text
10 dimensions
```

The decoder is needed during training to provide a reconstruction objective. It is not used to calculate anomaly scores during inference.

---

# 4. Masked training

The model uses masked graph representation learning.

The current setting is:

```text
mask_rate = 0.5
```

This means that 50% of the nodes are randomly selected during each training step.

Their input features are replaced by a learnable mask token:

```python
self.mask_token = nn.Parameter(
    torch.zeros(1, in_dim)
)
```

The model must use graph context and neighboring information to reconstruct the original features of the masked nodes.

This encourages the encoder to learn structural and contextual representations rather than simply copying the input features.

---

# 5. Reconstruction loss

The model uses scaled cosine reconstruction loss:

```python
cos_sim = F.cosine_similarity(
    original,
    reconstructed,
    dim=-1
)

loss = (1.0 - cos_sim).pow(3).mean()
```

The loss is calculated only for masked nodes.

The purpose is to make the reconstructed feature vector point in the same direction as the original feature vector.

The exponent:

```text
gamma = 3
```

makes the loss emphasize difficult reconstruction examples more strongly.

---

# 6. Training configuration

The command-line test creates the model with:

```python
MAGICModel(
    in_dim=10,
    hidden_dim=64,
    num_layers=3,
    mask_rate=0.5,
    k_neighbors=10,
    dropout=0.1,
)
```

Training uses:

```text
Epochs: 30
Learning rate: 0.001
Weight decay: 0.0005
Batch graphs: 1
```

The training graphs are processed one at a time to avoid excessive memory use.

The code also:

- adds self-loops to the graphs;
- clips gradients to a maximum norm of 1.0;
- applies dropout inside the GAT layers.

Self-loops are important because GAT layers require each node to have at least one incoming edge.

---

# 7. Training-loss output

The reported loss is:

```text
Epoch 1/30  loss=0.441017
Epoch 5/30  loss=0.035937
Epoch 10/30 loss=0.033726
Epoch 15/30 loss=0.034603
Epoch 20/30 loss=0.033629
Epoch 25/30 loss=0.031661
Epoch 30/30 loss=0.030376
```

The final loss is:

```text
0.030376
```

The loss decreased significantly from:

```text
0.441017
```

to:

```text
0.030376
```

This indicates that the model learned to reconstruct the masked node features effectively on the benign training graphs.

The loss is not itself an attack-detection metric. A low reconstruction loss means the model learned the training representation well, but it does not directly indicate precision or recall on malicious nodes.

---

# 8. Building the benign embedding index

After training, the model switches to evaluation mode and encodes every node in every training graph without masking.

The output shows:

```text
Encoded train graph 0: 167,265 embeddings dim=202
Encoded train graph 1: 380,204 embeddings dim=202
Encoded train graph 2: 342,624 embeddings dim=202
Encoded train graph 3: 389,106 embeddings dim=202
```

All embeddings are combined:

```text
1,279,199 benign embeddings
```

Each embedding contains 202 values.

Before building the KNN index, the embeddings are standardized dimension by dimension:

```python
all_emb_norm = (
    all_emb - self._train_emb_mean
) / self._train_emb_std
```

This prevents dimensions with larger numerical scales from dominating the Euclidean distance calculation.

---

# 9. BallTree KNN index

The implementation uses scikit-learn’s `BallTree`:

```python
self._ball_tree = BallTree(
    all_emb_norm,
    leaf_size=40,
    metric='euclidean'
)
```

The BallTree stores all benign training embeddings and supports efficient nearest-neighbor queries.

The model uses:

```text
k_neighbors = 10
```

For every test node, it finds the ten closest benign training embeddings.

A naive comparison against every one of the 1,279,199 training nodes would be expensive. The BallTree makes this nearest-neighbor search more practical.

---

# 10. Training KNN distance

The model estimates the typical benign KNN distance using up to 5,000 randomly selected training embeddings.

The reported value is:

```text
Avg KNN distance (train): 0.006263
```

This is used as a normalization baseline for test anomaly scores.

The score formula is:

```python
score = avg_test_knn_distance / avg_train_knn_distance
```

Therefore:

```text
score ≈ 1
```

means the test node is approximately as close to benign embeddings as a typical training node.

A score substantially greater than 1 means the node is farther from the benign embedding distribution.

---

# 11. MAGIC inference

The test graph is encoded without masking:

```python
emb, _ = self.forward(
    test_graph,
    x,
    training=False
)
```

The test embeddings are normalized using the training mean and standard deviation.

The BallTree then finds the ten nearest benign training embeddings:

```python
dists, _ = self._ball_tree.query(
    emb_norm,
    k=self.k_neighbors
)
```

The model calculates the average distance to those neighbors:

```python
avg_dists = dists.mean(axis=1)
```

The final anomaly score is:

```python
scores = avg_dists / (
    self._avg_train_dist + 1e-8
)
```

Higher scores indicate that the test node is farther from the benign training distribution.

---

# 12. Score range

The output reports:

```text
Score range: 0.0000 – 2364.5458
mean=36.6376
```

This means:

- minimum anomaly score: `0.0000`;
- maximum anomaly score: `2364.5458`;
- average anomaly score: `36.6376`.

The scores are highly skewed.

Most nodes are likely relatively close to the benign embedding space, while a smaller number of nodes are extremely far away.

The large maximum occurs because the average benign KNN distance is very small:

```text
0.006263
```

Even a moderate absolute distance can become a very large normalized score after division by this small baseline.

The magnitude of the score is not important by itself. The ranking of nodes and the selected threshold are more important.

---

# 13. Threshold selection

The script searches for a threshold that maximizes F1-score:

```python
best_thresh, best_f1, _ = find_best_threshold(
    test_labels,
    scores,
    metric='f1'
)
```

The selected threshold is:

```text
143.3058
```

Nodes with:

```text
score >= 143.3058
```

are classified as malicious.

Nodes with:

```text
score < 143.3058
```

are classified as benign.

The best F1-score at this threshold is:

```text
0.9183
```

Because there is no labeled validation graph, the threshold is selected on the labeled test graph. This is consistent with the evaluation limitation shared by the MAGIC experiment.

---

# 14. Confusion matrix

The MAGIC result is:

```text
TP=25,302
FP=4,484
FN=17
TN=314,964
```

These values sum to the complete test graph:

```text
25,302 + 4,484 + 17 + 314,964 = 344,767
```

Interpretation:

- 25,302 malicious nodes were correctly detected;
- 4,484 benign nodes were incorrectly flagged;
- 17 malicious nodes were missed;
- 314,964 benign nodes were correctly classified.

---

# 15. Metric interpretation

## Precision

```text
Precision: 0.8495
```

Approximately 84.95% of flagged nodes were actually malicious.

The model generates more false positives than the Random Forest baseline.

---

## Recall

```text
Recall: 0.9993
```

MAGIC detects approximately 99.93% of all malicious nodes.

Only 17 malicious nodes were missed.

This is the strongest aspect of the current MAGIC result.

---

## F1-score

```text
F1-score: 0.9183
```

The F1-score is lower than the Random Forest baseline because MAGIC has lower precision, despite having higher recall.

---

## False-positive rate

```text
FPR: 0.0140
```

Approximately 1.4% of benign nodes were incorrectly flagged.

This produces:

```text
4,484 false positives
```

That is higher than the Random Forest result:

```text
Random Forest FP: 1,702
MAGIC FP:         4,484
```

---

## ROC-AUC

```text
ROC-AUC: 0.9944
```

This is extremely strong ranking performance.

It means that MAGIC’s KNN-distance scores separate malicious nodes from benign nodes very effectively over all possible thresholds.

MAGIC has higher ROC-AUC than the Random Forest:

```text
MAGIC:         0.9944
Random Forest: 0.9708
```

---

## PR-AUC

```text
PR-AUC: 0.9307
```

This is also higher than the Random Forest result:

```text
MAGIC:         0.9307
Random Forest: 0.8807
```

Because the data is imbalanced, PR-AUC is especially important. MAGIC provides better malicious-node ranking than the Random Forest according to this metric.

---

# 16. MAGIC versus Random Forest

| Metric | Random Forest | MAGIC |
|---|---:|---:|
| Precision | 0.9350 | 0.8495 |
| Recall | 0.9673 | 0.9993 |
| F1-score | 0.9509 | 0.9183 |
| FPR | 0.0053 | 0.0140 |
| ROC-AUC | 0.9708 | 0.9944 |
| PR-AUC | 0.8807 | 0.9307 |
| TP | 24,492 | 25,302 |
| FP | 1,702 | 4,484 |
| FN | 827 | 17 |
| TN | 317,746 | 314,964 |

## Main conclusion

MAGIC produces better anomaly ranking:

```text
higher ROC-AUC
higher PR-AUC
```

MAGIC also detects more malicious nodes:

```text
Recall = 0.9993
```

However, the Random Forest produces fewer false positives and achieves a higher F1-score at its selected threshold:

```text
Random Forest F1 = 0.9509
MAGIC F1         = 0.9183
```

Therefore:

- if maximum attack detection is the priority, MAGIC is stronger;
- if reducing false alarms is the priority, the Random Forest is stronger at the current thresholds;
- if comparing ranking quality, MAGIC is currently better.

---

# 17. Threshold trade-off

The results show a typical security trade-off.

MAGIC:

```text
Very high recall
More false positives
```

Random Forest:

```text
Slightly lower recall
Fewer false positives
Higher F1-score
```

The choice depends on the system objective.

If missed attacks are unacceptable, MAGIC’s threshold may be appropriate.

If analyst workload and false alarms are more important, MAGIC’s threshold could be increased to improve precision and reduce FPR.

Because the score ranking is strong, MAGIC may support better operating points at other thresholds.

The current threshold maximizes F1, but a practical deployment could instead choose a threshold that satisfies a requirement such as:

```text
Recall >= 0.99
```

or:

```text
FPR <= 0.005
```

---

# 18. Model saving

At the end of the script, MAGIC is saved to:

```text
./models/saved/magic_model.pkl
```

The saved state includes:

- model weights;
- BallTree index;
- training embedding mean;
- training embedding standard deviation;
- average training KNN distance;
- fitted status;
- model configuration.

This allows the model to be loaded later without retraining:

```python
model = MAGICModel()
model.load('./models/saved/magic_model.pkl')
```

The saved model can then be used for inference on another compatible graph.

The threshold is not stored in the saved model. The selected threshold should be recorded separately if it is needed for later predictions:

```text
MAGIC threshold: 143.3058
```

---

# 19. Important evaluation limitation

The threshold was selected using:

```python
test_labels
```

because the dataset contains no labeled validation graph.

This means:

- ROC-AUC and PR-AUC evaluate the full score ranking;
- precision, recall, F1-score, FPR, and confusion-matrix values correspond to the test-calibrated threshold.

This protocol is appropriate for comparison with the original MAGIC setting, but it should be clearly documented in the final report.

A suitable description is:

> Since the dataset does not provide a separate labeled validation set, the anomaly threshold was calibrated on the labeled test graph, following the evaluation protocol used by the original MAGIC study. ROC-AUC and PR-AUC measure threshold-independent ranking quality, while the remaining classification metrics describe performance at the test-calibrated operating point.

---

# 20. Overall assessment

The current MAGIC implementation is working successfully:

```text
Training completed
Embeddings generated
BallTree constructed
Test inference completed
Threshold selected
Metrics calculated
Model saved
```

Its current strengths are:

- excellent ROC-AUC;
- excellent PR-AUC;
- nearly perfect recall;
- only 17 missed malicious nodes;
- graph-aware embeddings;
- benign-only representation learning;
- efficient KNN search using BallTree.

Its main weakness is the number of false positives:

```text
4,484 benign nodes flagged
```

Compared with the Random Forest baseline:

```text
Random Forest: 1,702 false positives
MAGIC:         4,484 false positives
```

The current project result can be summarized as:

> MAGIC provides the strongest anomaly ranking and highest malicious-node recall, while the Random Forest baseline provides better precision, lower false-positive rate, and higher F1-score at the currently selected thresholds.