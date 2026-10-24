"""
models/baseline_rf.py
=====================
Random Forest baseline model for node-level APT detection.

This is the simplest of the three models — no GNN, no graph structure,
just hand-crafted per-node features fed into a Random Forest classifier.

Its purpose in this project is to serve as a benchmark:
- If MAGIC and FLASH don't clearly beat this, something is wrong
- It shows the value of full GNN message-passing over hand-crafted features

This is a non-GNN baseline. It does not perform message passing, but it
does use hand-crafted node and local structural features derived from the
graph — degree counts and edge-type histograms give it a limited view of
each node's immediate neighbourhood without propagating information further.

Feature engineering (what replaces the GNN here):
- Raw node attributes        (5 dims  — from ndata['attr'])
- One-hot node type          (5 dims  — from ndata['type'])
- In-degree                  (1 dim   — how many edges point TO this node)
- Out-degree                 (1 dim   — how many edges point FROM this node)
- Total degree               (1 dim   — in + out)
- Edge type distribution     (10 dims — histogram of incoming edge types)
─────────────────────────────────────────────────────────────
Total input features: 23 dims per node

Anomaly score: normalised entropy of the RF node-type probability
distribution. Low entropy = confident type prediction = likely benign.
High entropy = uncertain = node does not resemble any known benign type.

Threshold note: the threshold is selected by searching over the test labels
directly (no separate validation set exists). Precision/recall/F1 should
therefore be treated as indicative. ROC-AUC and PR-AUC are the primary
metrics as they are threshold-independent.

Usage:
    from models.baseline_rf import RandomForestBaseline

    model = RandomForestBaseline()
    X_train, _ = model.extract_features(train_graph)
    X_test, _  = model.extract_features(test_graph)

    model.fit(X_train)
    scores, preds = model.predict(X_test)
"""

import torch
import numpy as np
import dgl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
import os


# Constants (must match data_loader.py) ────────────────────────────────────

NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 10


# Main class ────────────────────────────────────────────────────────────────

class RandomForestBaseline:
    """
    Random Forest baseline for node-level anomaly detection.

    Training (on benign graphs only):
      1. Extract hand-crafted features for every node
      2. Fit a StandardScaler to normalise features
      3. Train a Random Forest on node-type classification
         (same strategy as FLASH — misclassified nodes are anomalies)

    Inference (on test graph):
      1. Extract features
      2. Scale with the fitted scaler
      3. Get class probabilities from Random Forest
      4. Anomaly score = normalised entropy of node-type probability dist.
         (high entropy = uncertain = node doesn't fit learned benign patterns)
      5. Threshold scores to produce binary predictions
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        n_jobs: int = -1,
        random_state: int = 42
    ):
        """
        Parameters
        ----------
        n_estimators  : number of trees in the forest
        max_depth     : maximum depth per tree.
                        Keep this low (6-10) to prevent overfitting —
                        a perfect training accuracy means the RF memorised
                        the data and will produce overconfident scores.
        n_jobs        : parallel jobs (-1 = use all CPU cores)
        random_state  : seed for reproducibility
        """
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=50,      # each leaf must have 50+ samples
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight='balanced'   # handles class imbalance within types
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_dim = None

    # Feature extraction ────────────────────────────────────────────────────

    def extract_features(
        self,
        graph: dgl.DGLGraph
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a fixed-size feature vector for every node in the graph.

        Parameters
        ----------
        graph : DGLGraph

        Returns
        -------
        X : np.ndarray, shape (num_nodes, 23)
            Feature matrix — one row per node.
        node_types : np.ndarray, shape (num_nodes,)
            Integer node type labels (used as training targets).
        """
        num_nodes = graph.num_nodes()

        # 1. Raw node attributes (5 dims, binary) 
        node_attr = graph.ndata['attr'].numpy()                   # (N, 5)

        # 2. One-hot node type (5 dims) 
        node_type = graph.ndata['type'].numpy()                   # (N,)
        type_onehot = np.zeros((num_nodes, NUM_NODE_TYPES))
        type_onehot[np.arange(num_nodes), node_type] = 1.0       # (N, 5)

        # 3. Degree features (3 dims)
        in_deg  = graph.in_degrees().numpy().reshape(-1, 1)       # (N, 1)
        out_deg = graph.out_degrees().numpy().reshape(-1, 1)      # (N, 1)
        tot_deg = in_deg + out_deg                                 # (N, 1)

        # 4. Incoming edge type histogram (10 dims) 
        # For each node: count how many incoming edges of each type it has
        # This is a lightweight structural feature without full GNN message passing
        edge_type_hist = self._build_edge_type_histogram(graph)   # (N, 10)

        # Concatenate all features
        X = np.concatenate([
            node_attr,        # 5 dims
            type_onehot,      # 5 dims
            in_deg,           # 1 dim
            out_deg,          # 1 dim
            tot_deg,          # 1 dim
            edge_type_hist,   # 10 dims
        ], axis=1)            # → (N, 23)

        return X, node_type

    def _build_edge_type_histogram(self, graph: dgl.DGLGraph) -> np.ndarray:
        """
        For each node, build a histogram of incoming edge types.
        Shape: (num_nodes, NUM_EDGE_TYPES)

        This gives the RF a rough sense of what kind of interactions
        a node participates in, without doing full graph message passing.
        """
        num_nodes  = graph.num_nodes()
        edge_types = graph.edata['type'].numpy()   # (num_edges,)
        _, dst     = graph.edges()                 # dst nodes of each edge
        dst        = dst.numpy()

        hist = np.zeros((num_nodes, NUM_EDGE_TYPES), dtype=np.float32)
        for etype in range(NUM_EDGE_TYPES):
            mask      = (edge_types == etype)
            dst_nodes = dst[mask]
            # Count occurrences using bincount
            counts = np.bincount(dst_nodes, minlength=num_nodes)
            hist[:, etype] = counts[:num_nodes]

        # Normalise each row by total incoming edges (avoid scale differences)
        row_sums = hist.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0             # avoid divide-by-zero
        hist = hist / row_sums

        return hist

    # Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_graphs: list,
        verbose: bool = True
    ) -> None:
        """
        Fit the Random Forest on benign training graphs.

        Parameters
        ----------
        train_graphs : list of DGLGraph (benign only)
        verbose      : print progress
        """
        if verbose:
            print('Extracting features from training graphs...')

        all_X = []
        all_y = []

        for i, graph in enumerate(train_graphs):
            X, node_types = self.extract_features(graph)
            all_X.append(X)
            all_y.append(node_types)
            if verbose:
                print(f'  train{i}: {X.shape[0]:,} nodes  '
                      f'feature shape {X.shape}')

        X_train = np.vstack(all_X)     # (total_train_nodes, 23)
        y_train = np.concatenate(all_y) # (total_train_nodes,)

        if verbose:
            print(f'\nTotal training nodes : {len(X_train):,}')
            print(f'Feature dimensions   : {X_train.shape[1]}')
            unique, counts = np.unique(y_train, return_counts=True)
            print('Node type distribution in training data:')
            for t, c in zip(unique, counts):
                print(f'  type {t} : {c:,} nodes  ({100*c/len(y_train):.1f}%)')

        # Fit scaler on training data
        if verbose:
            print('\nFitting scaler...')
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Fit Random Forest
        if verbose:
            print('Training Random Forest...')
        self.rf.fit(X_train_scaled, y_train)

        self.is_fitted = True
        self.feature_dim = X_train.shape[1]

        if verbose:
            # Node-type classification accuracy on training data.
            # NOTE: one-hot node type is included as an input feature so
            # perfect accuracy here is expected and does not indicate
            # overfitting to attack labels — the RF has never seen any.
            train_preds = self.rf.predict(X_train_scaled)
            train_acc = (train_preds == y_train).mean()
            print(f'Training node-type accuracy : {train_acc:.4f} '
                  f'(expected ~1.0 — type is an input feature)')
            print('Random Forest fitted  ✓')

    # Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        test_graph: dgl.DGLGraph,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce anomaly scores and binary predictions for every node
        in the test graph.

        Anomaly score = normalised entropy of class probability distribution.
        Low entropy (confident) = benign. High entropy (uncertain) = anomalous.
        A node is flagged as malicious if its score >= threshold.

        Parameters
        ----------
        test_graph : DGLGraph
        threshold  : float or None
            If None, returns only scores (use find_best_threshold from
            metrics.py to choose a threshold, then call predict again).

        Returns
        -------
        scores : FloatTensor (num_nodes,)
            Anomaly score per node. Higher = more suspicious.
        preds  : LongTensor (num_nodes,)
            Binary predictions (0=benign, 1=malicious).
            All zeros if threshold is None.
        """
        if not self.is_fitted:
            raise RuntimeError('Call fit() before predict().')

        X_test, _ = self.extract_features(test_graph)
        X_test_scaled = self.scaler.transform(X_test)

        # Probability for each node type
        proba = self.rf.predict_proba(X_test_scaled)   # (N, num_types)

        # Calculate normalized entropy across the complete node-type probability distribution. 
        # Low entropy indicates confident classification; high entropy indicates uncertainty. 
        # A node confidently classified as its expected type scores low (benign). 
        # A node that doesn't fit any type cleanly scores high (suspicious).

        # We use entropy of the probability distribution as the score:
        # - Low entropy = confident prediction  = likely benign
        # - High entropy = uncertain prediction = likely anomalous
        # Clip to avoid log(0)
        proba_clipped = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(proba_clipped * np.log(proba_clipped), axis=1)
        # Normalise to [0, 1] range using max possible entropy
        max_entropy = np.log(proba.shape[1])
        scores = entropy / max_entropy                 # (N,)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        # Binary predictions
        if threshold is not None:
            preds = (scores_tensor >= threshold).long()
        else:
            preds = torch.zeros(len(scores), dtype=torch.long)

        return scores_tensor, preds

    # Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'rf': self.rf, 'scaler': self.scaler,
                         'is_fitted': self.is_fitted,
                         'feature_dim': self.feature_dim}, f)
        print(f'Model saved to {path}')

    def load(self, path: str) -> None:
        """Load a previously saved model from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.rf          = state['rf']
        self.scaler      = state['scaler']
        self.is_fitted   = state['is_fitted']
        self.feature_dim = state['feature_dim']
        print(f'Model loaded from {path}')


# Quick test 

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.data_loader import load_dataset, print_dataset_summary
    from utils.metrics import evaluate, print_results, find_best_threshold

    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'

    # Load data 
    print('Loading dataset...')
    train_graphs, test_graph, test_labels = load_dataset(data_dir)

    # Train
    model = RandomForestBaseline(n_estimators=200, max_depth=8)
    model.fit(train_graphs, verbose=True)

    # Predict
    print('\nRunning inference on test graph...')
    scores, _ = model.predict(test_graph, threshold=None)

    # Find best threshold (maximise F1)
    print('Searching for best threshold...')
    best_thresh, best_f1, _ = find_best_threshold(
        test_labels, scores, metric='f1'
    )
    print(f'Best threshold : {best_thresh:.4f}  →  F1 = {best_f1:.4f}')

    # Final predictions with best threshold
    _, preds = model.predict(test_graph, threshold=best_thresh)

    # Evaluate
    results = evaluate(test_labels, preds, scores,
                       model_name='Random Forest Baseline')
    print_results(results)

    print('\nbaseline_rf.py  ✓  done')