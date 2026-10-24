"""
utils/data_loader.py
====================
Loads the DARPA E3-THEIA dataset from DGL-format .pkl files.

What this file does:
1. Loads each .pkl file as a DGL graph
2. Reads malicious node IDs from metadata.json
3. Builds a label tensor for the test graph (0=benign, 1=malicious)
4. Provides a helper to extract node+edge features as plain tensors

Usage:
    from utils.data_loader import load_dataset

    train_graphs, test_graph, test_labels = load_dataset('./data')
"""

import os
import json
import pickle
import torch
import dgl
from typing import List, Tuple


# Constants ────────────────────────────────────────────────────────────────

NODE_FEATURE_DIM = 5    # from metadata.json: node_feature_dim
EDGE_FEATURE_DIM = 17   # from metadata.json: edge_feature_dim
NUM_NODE_TYPES   = 5    # types 0–4
NUM_EDGE_TYPES   = 10   # types 0–9


# Core loader ──────────────────────────────────────────────────────────────

def load_dataset(
    data_dir: str = './data'
) -> Tuple[List[dgl.DGLGraph], dgl.DGLGraph, torch.Tensor]:
    """
    Load all training graphs and the test graph.

    Parameters
    ----------
    data_dir : str
        Path to the folder containing train0.pkl … train3.pkl,
        test0.pkl, and metadata.json.

    Returns
    -------
    train_graphs : list of DGLGraph
        Four benign-only graphs used for training (no attack labels).
    test_graph : DGLGraph
        One graph containing both benign and malicious nodes.
    test_labels : torch.LongTensor, shape (num_test_nodes,)
        Per-node labels for the test graph: 0 = benign, 1 = malicious.
    """
    data_dir = os.path.abspath(data_dir)
    _check_files_exist(data_dir)

    # Load metadata (contains malicious node ID lists)
    meta = _load_metadata(data_dir)

    # Load training graphs (benign only — no labels needed)
    print('Loading training graphs...')
    train_graphs = []
    for i in range(meta['n_train']):
        path = os.path.join(data_dir, f'train{i}.pkl')
        g = _load_pkl(path)
        train_graphs.append(g)
        print(f'  train{i}.pkl  →  {g.num_nodes():>8,} nodes  '
              f'{g.num_edges():>9,} edges')

    # Load test graph
    print('\nLoading test graph...')
    test_path = os.path.join(data_dir, 'test0.pkl')
    test_graph = _load_pkl(test_path)
    print(f'  test0.pkl    →  {test_graph.num_nodes():>8,} nodes  '
          f'{test_graph.num_edges():>9,} edges')

    # Build label tensor for the test graph
    test_labels = _build_label_tensor(
        num_nodes=test_graph.num_nodes(),
        malicious_ids=meta['malicious'][0]   # malicious[1] is all 'NA'
    )
    n_mal = test_labels.sum().item()
    n_ben = (test_labels == 0).sum().item()
    print(f'\nTest labels:  {n_ben:,} benign  |  {int(n_mal):,} malicious  '
          f'({100 * n_mal / len(test_labels):.1f}% malicious)')

    return train_graphs, test_graph, test_labels


# Feature extraction ───────────────────────────────────────────────────────

def get_node_features(graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Return the node feature matrix.

    Shape: (num_nodes, 5)
    Values are binary (0.0 or 1.0) — one-hot encoded node type attributes.
    """
    return graph.ndata['attr'].float()


def get_edge_features(graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Return the edge feature matrix.

    Shape: (num_edges, 17)
    """
    return graph.edata['attr'].float()


def get_node_types(graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Return integer node type labels.

    Shape: (num_nodes,)
    Values: 0, 1, 2, 3, or 4
    These represent entity categories (process, file, socket, etc.)
    Used by FLASH as classification targets.
    """
    return graph.ndata['type'].long()


def get_edge_types(graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Return integer edge type labels.

    Shape: (num_edges,)
    Values: 0–9  (read, write, execute, connect, etc.)
    """
    return graph.edata['type'].long()


def get_combined_node_features(graph: dgl.DGLGraph) -> torch.Tensor:
    """
    Concatenate node attr (5-dim) with one-hot node type (5-dim)
    to give a richer 10-dim input feature vector.

    Shape: (num_nodes, 10)

    Why: the 5 attr features are binary but don't encode type directly.
    Adding the one-hot type gives the GNN more signal to work with.
    """
    attr      = get_node_features(graph)                          # (N, 5)
    node_type = get_node_types(graph)                             # (N,)
    type_onehot = torch.zeros(graph.num_nodes(), NUM_NODE_TYPES)  # (N, 5)
    type_onehot.scatter_(1, node_type.unsqueeze(1), 1.0)
    return torch.cat([attr, type_onehot], dim=1)                  # (N, 10)


# Dataset statistics ───────────────────────────────────────────────────────

def print_dataset_summary(
    train_graphs: List[dgl.DGLGraph],
    test_graph: dgl.DGLGraph,
    test_labels: torch.Tensor
) -> None:
    """Print a human-readable summary of the loaded dataset."""
    print('=' * 55)
    print('DATASET SUMMARY')
    print('=' * 55)

    total_train_nodes = sum(g.num_nodes() for g in train_graphs)
    total_train_edges = sum(g.num_edges() for g in train_graphs)

    print(f'Training graphs      : {len(train_graphs)}')
    print(f'  Total nodes        : {total_train_nodes:,}')
    print(f'  Total edges        : {total_train_edges:,}')
    print(f'  Labels             : benign only (no attack labels)')
    print()
    print(f'Test graph           : 1')
    print(f'  Total nodes        : {test_graph.num_nodes():,}')
    print(f'  Total edges        : {test_graph.num_edges():,}')

    n_mal = int(test_labels.sum().item())
    n_ben = int((test_labels == 0).sum().item())
    pct   = 100 * n_mal / len(test_labels)
    print(f'  Benign nodes       : {n_ben:,}')
    print(f'  Malicious nodes    : {n_mal:,}  ({pct:.1f}%)')
    print()
    print(f'Node feature dim     : {NODE_FEATURE_DIM}')
    print(f'Edge feature dim     : {EDGE_FEATURE_DIM}')
    print(f'Node types           : {NUM_NODE_TYPES}  (0–{NUM_NODE_TYPES-1})')
    print(f'Edge types           : {NUM_EDGE_TYPES}  (0–{NUM_EDGE_TYPES-1})')
    print('=' * 55)

    # Node type breakdown in test graph
    node_types = get_node_types(test_graph)
    print('Node type breakdown (test graph):')
    for t in range(NUM_NODE_TYPES):
        count = (node_types == t).sum().item()
        # Count how many of this type are malicious
        mal_mask = test_labels.bool()
        mal_of_type = ((node_types == t) & mal_mask).sum().item()
        print(f'  type {t} : {count:>8,} nodes   '
              f'({mal_of_type:,} malicious)')
    print('=' * 55)


# Internal helpers ─────────────────────────────────────────────────────────

def _load_pkl(path: str) -> dgl.DGLGraph:
    """Load a single .pkl file and return the DGL graph inside."""
    with open(path, 'rb') as f:
        graph = pickle.load(f)
    if not isinstance(graph, dgl.DGLGraph):
        raise TypeError(
            f'Expected a DGLGraph in {path}, got {type(graph)}. '
            'Check that your .pkl files are not lists.'
        )
    return graph


def _load_metadata(data_dir: str) -> dict:
    """Load and validate metadata.json."""
    meta_path = os.path.join(data_dir, 'metadata.json')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    required_keys = ['node_feature_dim', 'edge_feature_dim',
                     'malicious', 'n_train', 'n_test']
    for key in required_keys:
        if key not in meta:
            raise KeyError(f'metadata.json is missing key: "{key}"')

    return meta


def _build_label_tensor(
    num_nodes: int,
    malicious_ids: list
) -> torch.Tensor:
    """
    Build a (num_nodes,) long tensor of 0s and 1s.

    Parameters
    ----------
    num_nodes     : total number of nodes in the graph
    malicious_ids : list of integer node IDs that are malicious.
                    Any 'NA' entries are silently skipped.
    """
    labels = torch.zeros(num_nodes, dtype=torch.long)

    valid_ids = [nid for nid in malicious_ids
                 if nid != 'NA' and isinstance(nid, int)]

    if len(valid_ids) == 0:
        print('  WARNING: no valid malicious node IDs found — '
              'all labels set to 0 (benign)')
        return labels

    # Clamp IDs to valid range (safety check)
    valid_ids = [nid for nid in valid_ids if 0 <= nid < num_nodes]
    id_tensor = torch.tensor(valid_ids, dtype=torch.long)
    labels[id_tensor] = 1

    return labels


def _check_files_exist(data_dir: str) -> None:
    """Raise a clear error if expected files are missing."""
    required = [
        'metadata.json',
        'test0.pkl',
        'train0.pkl', 'train1.pkl', 'train2.pkl', 'train3.pkl'
    ]
    missing = [f for f in required
               if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            f'Missing files in {data_dir}:\n  ' +
            '\n  '.join(missing) +
            '\nMake sure DATA_DIR points to the folder containing '
            'train0.pkl … train3.pkl, test0.pkl, and metadata.json.'
        )


# Quick test

if __name__ == '__main__':
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'
    print(f'Testing data_loader with data_dir = "{data_dir}"\n')

    train_graphs, test_graph, test_labels = load_dataset(data_dir)
    print()
    print_dataset_summary(train_graphs, test_graph, test_labels)

    # Verify feature extraction works
    print('\nVerifying feature extraction...')
    x     = get_node_features(train_graphs[0])
    x_ext = get_combined_node_features(train_graphs[0])
    nt    = get_node_types(train_graphs[0])
    et    = get_edge_types(train_graphs[0])
    print(f'  get_node_features()          → shape {tuple(x.shape)}')
    print(f'  get_combined_node_features() → shape {tuple(x_ext.shape)}')
    print(f'  get_node_types()             → shape {tuple(nt.shape)}')
    print(f'  get_edge_types()             → shape {tuple(et.shape)}')
    print('\ndata_loader.py  ✓  all checks passed')