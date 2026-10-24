## Detailed Description of `utils/data_loader.py`

The file `data_loader.py` is the project’s central data-loading and preprocessing module. It loads the DARPA E3-THEIA provenance graphs, validates the required files, creates test labels, extracts node and edge features, and prints dataset statistics.

It is used by the model-training and evaluation code to ensure that all graphs are loaded consistently.

### 1. Purpose of the file
The module performs five main tasks:
1. **Loads the dataset files**
   - `train0.pkl`
   - `train1.pkl`
   - `train2.pkl`
   - `train3.pkl`
   - `test0.pkl`
   - `metadata.json`
2. **Validates the input directory**
   - Checks whether all required files exist.
   - Raises a clear error if any file is missing.
3. **Builds test labels**
   - Reads malicious node IDs from `metadata.json`.
   - Creates a tensor containing:
     - `0` for benign nodes;
     - `1` for malicious nodes.
4. **Extracts graph features**
   - Node attributes;
   - Edge attributes;
   - Node types;
   - Edge types;
   - Optional combined node features.
5. **Prints a complete dataset summary**
   - Number of graphs;
   - Number of nodes and edges;
   - Class distribution;
   - Feature dimensions;
   - Node-type distribution;
   - Validation of feature-extraction functions.

## 2. Required input files
The loader expects the following files inside the data directory:
```text
data/
├── metadata.json
├── train0.pkl
├── train1.pkl
├── train2.pkl
├── train3.pkl
└── test0.pkl
```
The default directory is:
```python
"./data"
```
The directory can also be provided manually when running the file from the terminal.
For example:
```bash
python utils/data_loader.py ./data
```
or:
```bash
python utils/data_loader.py /absolute/path/to/data
```
If no argument is supplied, this code is used:
```python
data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'
```
Therefore, both of these commands are valid:
```bash
python utils/data_loader.py
python utils/data_loader.py ./data
```

## 3. Constants
The file defines several dataset constants:
```python
NODE_FEATURE_DIM = 5
EDGE_FEATURE_DIM = 17
NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 10
```
Their meaning is:
| Constant | Meaning |
|---|---|
| `NODE_FEATURE_DIM` | Number of features stored for every node |
| `EDGE_FEATURE_DIM` | Number of features stored for every edge |
| `NUM_NODE_TYPES` | Number of supported node types |
| `NUM_EDGE_TYPES` | Number of supported edge types |

The observed dataset values are:
```text
Node feature dimension: 5
Edge feature dimension: 17
Node types: 0–4
Edge types: 0–9
```
The node feature and edge feature dimensions agree with `metadata.json`.
One important detail is that the edge feature dimension and edge type count are different concepts:
- each edge has **17 numerical attributes**;
- each edge has one categorical **edge type** from 0 to 9.

Similarly:
- each node has **5 numerical attributes**;
- each node has one categorical **node type** from 0 to 4.

## 4. Main function: `load_dataset`
The main public function is:
```python
load_dataset(data_dir='./data')
```
Its signature is:
```python
def load_dataset(
    data_dir: str = './data'
) -> Tuple[List[dgl.DGLGraph], dgl.DGLGraph, torch.Tensor]:
```
It returns three objects:
```python
train_graphs, test_graph, test_labels
```
### Returned values
**`train_graphs`**

A list containing four DGL graphs:
```python
[
    train0_graph,
    train1_graph,
    train2_graph,
    train3_graph
]
```
The terminal output shows:
```text
train0.pkl: 167,265 nodes, 284,562 edges
train1.pkl: 380,204 nodes, 674,723 edges
train2.pkl: 342,624 nodes, 587,111 edges
train3.pkl: 389,106 nodes, 700,318 edges
```
Combined training-set size:
```text
1,279,199 nodes
2,246,714 edges
```
The training graphs are treated as benign-only graphs by this loader. No attack labels are created for them.

**`test_graph`**

The graph loaded from:
```text
test0.pkl
```
It contains:
```text
344,767 nodes
628,107 edges
```
This graph contains both benign and malicious nodes.

**`test_labels`**

A one-dimensional `torch.LongTensor` with one label per test node:
```text
shape: (344,767,)
```
The label format is:
```text
0 = benign
1 = malicious
```
The observed test distribution is:
```text
319,448 benign nodes
25,319 malicious nodes
```
The malicious percentage is:
```text
25,319 / 344,767 = approximately 7.3%
```

### 5. Dataset loading process
The function first converts the input path into an absolute path:
```python
data_dir = os.path.abspath(data_dir)
```
This avoids problems caused by running the script from different working directories.

It then checks that the expected files exist:
```python
_check_files_exist(data_dir)
```
Next, it loads the metadata:
```python
meta = _load_metadata(data_dir)
```
The training graphs are loaded in a loop:
```python
for i in range(meta['n_train']):
    path = os.path.join(data_dir, f'train{i}.pkl')
    g = _load_pkl(path)
    train_graphs.append(g)
```
The number of training graphs is not hardcoded in this loop. It is read from:
```python
meta['n_train']
```
The test graph is loaded separately:
```python
test_path = os.path.join(data_dir, 'test0.pkl')
test_graph = _load_pkl(test_path)
```
Finally, the label tensor is created using the malicious node IDs from:
```python
meta['malicious'][0]
```
The second metadata list, `meta['malicious'][1]`, is not used because it contains `"NA"` values according to the current dataset inspection.

### 6. Loading serialized DGL graphs
The private helper `_load_pkl` loads one pickle file:
```python
def _load_pkl(path: str) -> dgl.DGLGraph:
```
Its implementation is:
```python
with open(path, 'rb') as f:
    graph = pickle.load(f)
```
After loading, it verifies that the result is a DGL graph:
```python
if not isinstance(graph, dgl.DGLGraph):
    raise TypeError(...)
```
This validation prevents an unclear error later if the pickle file contains a list, dictionary, or another object instead of a graph.

The graph is expected to have fields such as:
```python
graph.ndata['attr']
graph.ndata['type']
graph.edata['attr']
graph.edata['type']
```

### 7. Building malicious labels
The helper function is:
```python
_build_label_tensor(
    num_nodes,
    malicious_ids
)
```
It creates an all-zero tensor:
```python
labels = torch.zeros(num_nodes, dtype=torch.long)
```
This initially assumes that every node is benign.

It then filters the metadata list:
```python
valid_ids = [
    nid for nid in malicious_ids
    if nid != 'NA' and isinstance(nid, int)
]
```
This removes:
- `"NA"` entries;
- non-integer values.

The valid node IDs are filtered again to make sure they are inside the graph range:
```python
valid_ids = [
    nid for nid in valid_ids
    if 0 <= nid < num_nodes
]
```
Finally, those node positions are set to one:
```python
labels[id_tensor] = 1
```
The result is a label tensor equivalent to:
```python
labels[node_id] = 1
```
for each malicious node ID.

### Important label interpretation
The current loader defines malicious nodes using the metadata list:
```python
metadata.json["malicious"][0]
```
It does **not** define malicious nodes using:
```python
graph.ndata["type"] == 4
```
This is important because the earlier graph inspection showed that type 4 contains only three nodes in `test0.pkl`, whereas the loader reports 25,319 malicious nodes.

The current output shows that the malicious nodes are distributed as follows:

| Node type | Total nodes | Malicious nodes |
|---:|---:|---:|
| 0 | 12,936 | 23 |
| 1 | 48,044 | 5 |
| 2 | 237,593 | 0 |
| 3 | 46,191 | 25,291 |
| 4 | 3 | 0 |

Therefore, the current project has two different concepts:
- **node type**: a categorical property stored in the graph;
- **malicious label**: a separate label created from metadata node IDs.

The malicious class is primarily associated with node type 3, not node type 4. Any model or documentation should use the metadata-generated labels if the goal is to reproduce the `data_loader.py` output.

### 8. Feature-extraction functions
**`get_node_features`**
```python
get_node_features(graph)
```
This returns:
```python
graph.ndata['attr'].float()
```
Expected shape:
```text
(num_nodes, 5)
```
For the first training graph:
```text
(167265, 5)
```
These are the original node attributes and are the standard five-dimensional GNN input.

Example:
```python
train_graphs, test_graph, test_labels = load_dataset('./data')

x = get_node_features(train_graphs[0])

print(x.shape)
# torch.Size([167265, 5])
```

**`get_edge_features`**
```python
get_edge_features(graph)
```
This returns:
```python
graph.edata['attr'].float()
```
Expected shape:
```text
(num_edges, 17)
```
For `train0.pkl`:
```text
(284562, 17)
```
Example:
```python
edge_features = get_edge_features(train_graphs[0])
print(edge_features.shape)
```
Edge features are currently extracted but are not automatically used by the basic data-loading process. A GNN model must explicitly accept and use them.

**`get_node_types`**
```python
get_node_types(graph)
```
This returns the categorical node type tensor:
```python
graph.ndata['type'].long()
```
Expected shape:
```text
(num_nodes,)
```
Example:
```python
node_types = get_node_types(test_graph)

print(node_types.shape)
# torch.Size([344767])

print(torch.unique(node_types))
# tensor([0, 1, 2, 3, 4])
```
These values are categorical identifiers, not malicious labels.

A node type of 3 does not automatically mean that every node is malicious. The output confirms that only 25,291 of the 46,191 type-3 nodes are malicious.

**`get_edge_types`**
```python
get_edge_types(graph)
```
This returns:
```python
graph.edata['type'].long()
```
Expected shape:
```text
(num_edges,)
```
For `train0.pkl`:
```text
(284562,)
```
Example:
```python
edge_types = get_edge_types(train_graphs[0])

print(edge_types.shape)
print(torch.unique(edge_types))
```
The edge types are categorical relationship labels. They should not be confused with the 17-dimensional edge attribute vectors.

### 9. Combined node features
The function:
```python
get_combined_node_features(graph)
```
creates a richer node input by concatenating:
1. the original five-dimensional node attributes;
2. a five-dimensional one-hot encoding of the node type.

The resulting shape is:
```text
(num_nodes, 10)
```
The implementation is:
```python
attr = get_node_features(graph)
node_type = get_node_types(graph)

type_onehot = torch.zeros(
    graph.num_nodes(),
    NUM_NODE_TYPES
)

type_onehot.scatter_(
    1,
    node_type.unsqueeze(1),
    1.0
)

return torch.cat([attr, type_onehot], dim=1)
```
For `train0.pkl`, the output is:
```text
(167265, 10)
```
Conceptually, a node may have an input like:
```text
Original attributes:
[1, 0, 0, 0, 0]

Node type:
2

One-hot node type:
[0, 0, 1, 0, 0]

Combined vector:
[1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
```
This allows the model to receive both:
- the original node attributes;
- explicit categorical node-type information.

## When to use combined features
Use `get_node_features` when the model expects:
```python
input_dimension = 5
```
Use `get_combined_node_features` when the model expects:
```python
input_dimension = 10
```
For example:
```python
x = get_combined_node_features(test_graph)
model = MyGNN(in_dim=10)
```
The model’s first layer must match the selected feature dimension. Passing ten-dimensional features to a model configured for five dimensions will cause a matrix-size error.

### 10. Dataset summary
The function:
```python
print_dataset_summary(
    train_graphs,
    test_graph,
    test_labels
)
```
prints a human-readable overview of the loaded dataset.

It calculates the total training size using:
```python
total_train_nodes = sum(
    g.num_nodes() for g in train_graphs
)
```
and:
```python
total_train_edges = sum(
    g.num_edges() for g in train_graphs
)
```
The summary reports:
```text
Training graphs: 4
Training nodes: 1,279,199
Training edges: 2,246,714
Training labels: benign only

Test nodes: 344,767
Test edges: 628,107
Benign nodes: 319,448
Malicious nodes: 25,319
```
It also prints the node-type breakdown for the test graph and counts how many malicious nodes belong to each type.

This is useful for detecting label or data problems. For example, the output reveals that:
- type 2 has no malicious nodes;
- type 3 contains nearly all malicious nodes;
- type 4 has no malicious nodes under the metadata-based labels.

### Command-line test mode
The bottom of the file contains:
```python
if __name__ == '__main__':
```
This means the module behaves differently depending on how it is used.

**When imported**
```python
from utils.data_loader import load_dataset
```
Only the functions and constants are made available. The dataset is not loaded automatically.

**When executed directly**
```bash
python utils/data_loader.py
```
The file:
1. loads the dataset;
2. prints graph sizes;
3. builds test labels;
4. prints the full dataset summary;
5. extracts sample features;
6. verifies the tensor shapes;
7. prints:
```text
data_loader.py  ✓  all checks passed
```
This direct-execution mode is a convenient smoke test after changing the dataset or loader.

### Typical usage in another Python file
A basic usage example is:
```python
from utils.data_loader import (
    load_dataset,
    get_node_features,
    get_edge_features,
    get_node_types,
    get_edge_types,
    get_combined_node_features,
    print_dataset_summary,
)

train_graphs, test_graph, test_labels = load_dataset(
    './data'
)

print_dataset_summary(
    train_graphs,
    test_graph,
    test_labels
)

train_graph = train_graphs[0]

node_features = get_node_features(train_graph)
edge_features = get_edge_features(train_graph)
node_types = get_node_types(train_graph)
edge_types = get_edge_types(train_graph)

print(node_features.shape)
print(edge_features.shape)
print(node_types.shape)
print(edge_types.shape)
```
For a model using extended features:
```python
x = get_combined_node_features(train_graph)

print(x.shape)
# torch.Size([167265, 10])
```

### How to use it for training
The training graphs can be loaded as follows:
```python
train_graphs, test_graph, test_labels = load_dataset('./data')
```
Then each graph can be processed independently:
```python
for graph in train_graphs:
    x = get_node_features(graph)

    logits = model(graph, x)
```
For test evaluation:
```python
x_test = get_node_features(test_graph)
logits = model(test_graph, x_test)

predictions = logits.argmax(dim=1)
```
For binary models that output probabilities:
```python
probabilities = torch.sigmoid(logits)
predictions = (probabilities >= 0.5).long()
```
The predictions should then be compared with:
```python
test_labels
```
Because the malicious class represents 7.3% of the test graph, the classification threshold may need to be tuned rather than fixed at 0.5.

### Fine-tuning and improvement recommendations
**A. Validate metadata dimensions instead of relying only on constants**

Currently, the constants are manually defined:
```python
NODE_FEATURE_DIM = 5
EDGE_FEATURE_DIM = 17
NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 10
```
A more robust loader would compare these constants with the actual metadata:
```python
if meta['node_feature_dim'] != NODE_FEATURE_DIM:
    raise ValueError(...)
```
It should also inspect the loaded graph:
```python
actual_node_dim = graph.ndata['attr'].shape[1]
actual_edge_dim = graph.edata['attr'].shape[1]
```
This prevents silent errors when the dataset changes.

**B. Validate all graph fields**

The loader currently verifies only that the pickle contains a `DGLGraph`. It does not explicitly verify that the required fields exist.

A stronger validation should check for:
```python
'attr' in graph.ndata
'type' in graph.ndata
'attr' in graph.edata
'type' in graph.edata
```
Without this validation, a graph with missing fields could load successfully and fail later inside a model.

**C. Validate tensor shapes and types**

The loader should verify that:
```text
ndata["attr"] has shape (N, 5)
ndata["type"] has shape (N,)
edata["attr"] has shape (E, 17)
edata["type"] has shape (E,)
```
It should also check that:
- node IDs are within range;
- node types are in `0–4`;
- edge types are in `0–9`;
- the number of labels equals the number of test nodes.

These checks would make dataset errors easier to diagnose.

**D. Warn when malicious IDs are outside the graph**

The current implementation silently discards malicious IDs outside the valid range:
```python
valid_ids = [
    nid for nid in valid_ids
    if 0 <= nid < num_nodes
]
```
This is safe, but it can hide a metadata mismatch.

A better implementation should report how many IDs were discarded:
```text
Warning: 12 malicious IDs were outside the test graph range.
```
This is especially important if metadata node IDs use a global indexing scheme while each pickle file uses local node IDs.

**E. Remove the “silent all-benign” behavior**

If no valid malicious IDs are found, the current code prints a warning and returns all-zero labels:
```python
if len(valid_ids) == 0:
    print('WARNING: no valid malicious node IDs found...')
    return labels
```
This can be dangerous because the pipeline continues with apparently valid labels even though the label construction failed.

For research experiments, a safer behavior would be to raise an exception unless the caller explicitly requests an all-benign fallback. Otherwise, a metadata error could lead to a misleading model evaluation.

**F. Read `n_test` consistently**

The metadata contains:
```python
n_test
```
but the current loader always loads:
```python
test0.pkl
```
A future version could validate that:
```python
meta['n_test'] == 1
```
or support multiple test graphs if the dataset later contains:
```text
test0.pkl
test1.pkl
...
```

**G. Resolve the label-definition ambiguity**

The most important conceptual issue is the difference between:
```python
metadata['malicious'][0]
```
and:
```python
graph.ndata['type'] == 4
```
The terminal output confirms that these produce different interpretations:
- metadata labels: 25,319 malicious nodes;
- node type 4: only 3 nodes in the test graph.

The project should explicitly choose one of the following:

**Metadata-based labels**

Use this if the attack ground truth is defined by the metadata:
```python
test_labels = _build_label_tensor(...)
```
This is the behavior currently implemented.

**Type-based labels**

Use this only if node type 4 is officially defined as the malicious class:
```python
test_labels = (
    test_graph.ndata['type'] == 4
).long()
```
These approaches should not be mixed. The selected label definition should be documented in the training and evaluation code.

**H. Add edge-feature support to the models**

`data_loader.py` extracts edge features, but extracting them does not automatically make the GNN use them.

A model must explicitly incorporate them during message passing. For example, the message function could use:
```python
src_node_features
edge_features
```
to calculate messages.

If the model only uses:
```python
get_node_features(graph)
```
then edge attributes and edge types are not contributing to the prediction, even though they are available.

**I. Consider memory usage**

The full dataset contains more than:
```text
1.27 million training nodes
344,767 test nodes
```
and millions of edges.

Functions such as:
```python
get_combined_node_features(graph)
```
allocate a new tensor. For very large graphs, this increases memory use.

Possible improvements include:
- processing one graph at a time;
- avoiding unnecessary copies;
- using sparse or compact type encodings;
- moving tensors to a selected device only when needed;
- sampling neighborhoods for training;
- using DGL minibatch or neighbor-sampling utilities.

**J. Avoid loading untrusted pickle files**

Python pickle loading can execute arbitrary code if the pickle file is malicious or untrusted:
```python
pickle.load(f)
```
This is acceptable for trusted local dataset files, but the loader should not be used with unknown pickle files without inspection.

### What the current output confirms
The successful terminal output confirms that:
- all required files were found;
- all four training graphs loaded successfully;
- the test graph loaded successfully;
- metadata labels were created;
- the label count matches the expected test graph size;
- feature extraction works;
- combined feature construction works;
- node-type extraction works;
- edge-type extraction works.

The specific validation shapes are:
```text
get_node_features()
→ (167265, 5)

get_combined_node_features()
→ (167265, 10)

get_node_types()
→ (167265,)

get_edge_types()
→ (284562,)
```
This means the loader is currently functioning correctly for the local dataset.

The most important thing to clarify before further model development is the label semantics: the current loader uses the 25,319 metadata-listed malicious nodes, while the graph’s type-4 category contains only three nodes. That distinction directly affects training targets, evaluation metrics, visualization, and the interpretation of intrusion-detection results.