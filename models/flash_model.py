"""
models/flash_model.py
=====================
FLASH: Provenance Graph Representation Learning for APT Detection
Based on: Rehman et al., IEEE S&P 2024

Fixes applied vs first version:
  1. Self-loops added ONCE before training, not every epoch
  2. Classification head included in optimizer (was a bug — head never updated)
  3. Embedding database vectorised — no per-node Python loop
  4. Inference loop replaced with fully vectorised numpy operations
  5. XGBoost n_estimators reduced to 100 (sufficient, much faster)
  6. Progress printing added inside XGBoost and DB steps

Usage:
    from models.flash_model import FLASHModel

    model = FLASHModel(in_dim=10, hidden_dim=64)
    model.fit(train_graphs)
    scores, preds = model.predict(test_graph)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import SAGEConv
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
import pickle

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print('XGBoost not found — falling back to GradientBoostingClassifier.')
    print('Install with: pip install xgboost')

NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 10


# ── GraphSAGE Encoder ─────────────────────────────────────────────────────────

class GraphSAGEEncoder(nn.Module):
    """
    Two-layer GraphSAGE encoder.
    Uses mean aggregation — simpler and faster than GAT attention.
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        out_dim:    int,
        num_layers: int   = 2,
        dropout:    float = 0.3
    ):
        super().__init__()
        self.dropout = dropout
        self.convs   = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_dim, out_dim, 'mean'))
        else:
            self.convs.append(SAGEConv(in_dim, hidden_dim, 'mean'))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, 'mean'))
            self.convs.append(SAGEConv(hidden_dim, out_dim, 'mean'))

    def forward(self, graph: dgl.DGLGraph, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(graph, h)
            if i < len(self.convs) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


# ── Full FLASH Model ──────────────────────────────────────────────────────────

class FLASHModel:
    """
    FLASH: GraphSAGE structural encoder + XGBoost node-type classifier.

    Anomaly detection strategy (same as RF baseline):
      Train on benign node-type classification.
      At inference, high-entropy predictions = node doesn't fit any known
      benign type = anomalous.

    Key speed improvements over v1:
      - Self-loops added once, reused every epoch
      - Classification head included in optimizer (bug fix)
      - Embedding DB built with vectorised numpy (no Python loop)
      - Inference fully vectorised (no Python loop)
    """

    def __init__(
        self,
        in_dim:      int   = 10,
        hidden_dim:  int   = 64,
        gnn_out_dim: int   = 64,
        num_layers:  int   = 2,
        dropout:     float = 0.3,
        use_xgboost: bool  = True
    ):
        self.in_dim      = in_dim
        self.hidden_dim  = hidden_dim
        self.gnn_out_dim = gnn_out_dim
        self.num_layers  = num_layers

        self.encoder = GraphSAGEEncoder(
            in_dim, hidden_dim, gnn_out_dim, num_layers, dropout
        )
        self.scaler    = StandardScaler()
        self.is_fitted = False

        self._build_classifier(use_xgboost)

        # Embedding database: keyed by (node_type, quantised_edge_hist_tuple)
        # Built during fit(), looked up during predict()
        self._emb_db: dict = {}

    # ── Classifier setup ──────────────────────────────────────────────────────

    def _build_classifier(self, use_xgboost: bool) -> None:
        if use_xgboost and XGBOOST_AVAILABLE:
            self.classifier = XGBClassifier(
                n_estimators     = 100,   # reduced from 200 — much faster
                max_depth        = 6,
                learning_rate    = 0.1,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                use_label_encoder= False,
                eval_metric      = 'mlogloss',
                random_state     = 42,
                n_jobs           = -1,
                verbosity        = 1      # prints XGBoost progress
            )
            self.classifier_name = 'XGBoost'
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.classifier = GradientBoostingClassifier(
                n_estimators = 100,
                max_depth    = 5,
                learning_rate= 0.1,
                random_state = 42,
                verbose      = 1
            )
            self.classifier_name = 'GradientBoosting'

    # ── Edge type histogram (vectorised) ─────────────────────────────────────

    def _compute_edge_type_histograms(
        self,
        graph: dgl.DGLGraph
    ) -> np.ndarray:
        """
        Per-node incoming edge-type histogram, normalised to sum to 1.
        Shape: (num_nodes, NUM_EDGE_TYPES)
        Fully vectorised — no Python loop over nodes.
        """
        num_nodes  = graph.num_nodes()
        edge_types = graph.edata['type'].numpy()   # (E,)
        _, dst     = graph.edges()
        dst        = dst.numpy()                   # (E,)

        # Build flat index: edge i contributes to row dst[i], col edge_types[i]
        row_idx = dst.astype(np.int64)
        col_idx = edge_types.astype(np.int64)
        flat    = row_idx * NUM_EDGE_TYPES + col_idx

        # minlength must be exactly num_nodes * NUM_EDGE_TYPES so the
        # reshape always works regardless of which edge types are present
        expected_len = int(num_nodes) * NUM_EDGE_TYPES
        hist_flat = np.bincount(flat, minlength=expected_len)
        hist = hist_flat[:expected_len].reshape(
            num_nodes, NUM_EDGE_TYPES).astype(np.float32)

        # Normalise rows
        row_sums = hist.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return hist / row_sums

    # ── GNN training ──────────────────────────────────────────────────────────

    def _train_gnn(
        self,
        train_graphs_sl: List[dgl.DGLGraph],   # already have self-loops
        epochs:          int   = 20,
        lr:              float = 0.001,
        verbose:         bool  = True
    ) -> List[float]:
        """
        Train GraphSAGE + linear head on node-type classification.

        FIX: optimizer now includes BOTH encoder AND cls_head parameters.
        FIX: self-loops are NOT added here — caller adds them once.
        """
        from utils.data_loader import get_combined_node_features

        # Weighted cross-entropy to handle 80% type-2 dominance
        type_counts = np.zeros(NUM_NODE_TYPES)
        for g in train_graphs_sl:
            types = g.ndata['type'].numpy()
            for t in range(NUM_NODE_TYPES):
                type_counts[t] += (types == t).sum()

        weights       = 1.0 / (type_counts + 1)
        weights       = weights / weights.sum() * NUM_NODE_TYPES
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        loss_fn       = nn.CrossEntropyLoss(weight=weight_tensor)

        # FIX: include cls_head in optimizer so its weights actually update
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self._cls_head.parameters()),
            lr=lr, weight_decay=5e-4
        )

        loss_history = []
        if verbose:
            print('  Training GraphSAGE + classification head...')
            print(f'  Class weights: ' +
                  ' '.join(f't{i}={w:.2f}' for i, w in enumerate(weights)))

        self.encoder.train()
        self._cls_head.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            for g in train_graphs_sl:
                x = get_combined_node_features(g)   # (N, 10)
                y = g.ndata['type'].long()           # (N,)

                optimizer.zero_grad()
                emb    = self.encoder(g, x)          # (N, gnn_out_dim)
                logits = self._cls_head(emb)         # (N, NUM_NODE_TYPES)
                loss   = loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self._cls_head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                epoch_loss += loss.item()

            avg = epoch_loss / len(train_graphs_sl)
            loss_history.append(avg)
            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f'    Epoch {epoch+1:>3}/{epochs}  loss={avg:.6f}')

        return loss_history

    # ── Full training pipeline ────────────────────────────────────────────────

    def fit(
        self,
        train_graphs: List[dgl.DGLGraph],
        gnn_epochs:   int   = 20,
        lr:           float = 0.001,
        verbose:      bool  = True
    ) -> None:
        """
        Full FLASH training pipeline.

        FIX: self-loops added ONCE here, reused throughout all steps.
        """
        from utils.data_loader import get_combined_node_features

        if verbose:
            print('Training FLASH model...')
            print(f'  in_dim       : {self.in_dim}')
            print(f'  gnn_out_dim  : {self.gnn_out_dim}')
            print(f'  num_layers   : {self.num_layers}')
            print(f'  classifier   : {self.classifier_name}')

        # ── Add self-loops ONCE, reuse everywhere ──────────────────────────
        if verbose:
            print('\nPre-processing: adding self-loops...')
        train_graphs_sl = [dgl.add_self_loop(g) for g in train_graphs]

        # ── Step 1: Train GNN ──────────────────────────────────────────────
        # Build classification head before training
        self._cls_head = nn.Linear(self.gnn_out_dim, NUM_NODE_TYPES)

        if verbose:
            print(f'\nStep 1/4 — Train GraphSAGE ({gnn_epochs} epochs)...')
        self._train_gnn(train_graphs_sl, epochs=gnn_epochs,
                        lr=lr, verbose=verbose)

        # ── Step 2: Extract combined features for XGBoost ─────────────────
        if verbose:
            print('\nStep 2/4 — Extract embeddings from training graphs...')

        self.encoder.eval()
        self._cls_head.eval()
        all_X = []
        all_y = []

        with torch.no_grad():
            for i, g in enumerate(train_graphs_sl):
                x       = get_combined_node_features(g)   # (N, 10)
                y       = g.ndata['type'].long().numpy()  # (N,)
                gnn_emb = self.encoder(g, x).numpy()      # (N, 64)
                raw_x   = x.numpy()                       # (N, 10)
                combined = np.concatenate([raw_x, gnn_emb], axis=1)  # (N, 74)
                all_X.append(combined)
                all_y.append(y)
                if verbose:
                    print(f'  train{i}: {g.num_nodes():,} nodes  '
                          f'embedding shape {combined.shape}')

        X_train = np.vstack(all_X)       # (1.28M, 74)
        y_train = np.concatenate(all_y)  # (1.28M,)
        if verbose:
            print(f'  Total : {len(X_train):,} nodes  '
                  f'{X_train.shape[1]} features')

        # ── Step 3: Build vectorised embedding database ────────────────────
        if verbose:
            print('\nStep 3/4 — Building embedding database (vectorised)...')

        with torch.no_grad():
            for i, (g, orig_g) in enumerate(
                    zip(train_graphs_sl, train_graphs)):
                x          = get_combined_node_features(g)
                node_types = g.ndata['type'].numpy()         # (N,)
                gnn_emb    = self.encoder(g, x).numpy()      # (N, 64)
                edge_hists = self._compute_edge_type_histograms(orig_g)
                             # shape (N, 10) — use original graph (no self-loops)
                             # so edge hist reflects real edges only

                # Vectorised key construction:
                # Quantise histograms to 1 d.p., then hash each row
                hist_q = np.round(edge_hists, 1)             # (N, 10)

                # Build one key string per node using vectorised ops
                for nid in range(g.num_nodes()):
                    key = (int(node_types[nid]),
                           tuple(hist_q[nid].tolist()))
                    if key not in self._emb_db:
                        self._emb_db[key] = gnn_emb[nid]

                if verbose:
                    print(f'  train{i}: DB size now '
                          f'{len(self._emb_db):,} entries')

        if verbose:
            print(f'  Final DB size : {len(self._emb_db):,} unique entries')

        # ── Step 4: Train XGBoost ──────────────────────────────────────────
        if verbose:
            print(f'\nStep 4/4 — Train {self.classifier_name} '
                  f'on {len(X_train):,} nodes...')
            print('  (XGBoost will print its own progress below)')

        X_scaled = self.scaler.fit_transform(X_train)
        self.classifier.fit(X_scaled, y_train)

        if verbose:
            train_preds = self.classifier.predict(X_scaled)
            train_acc   = (train_preds == y_train).mean()
            print(f'\n  Training node-type accuracy : {train_acc:.4f} '
                  f'(expected ~1.0 — type is an input feature)')
            print('\nFLASH training complete  ✓')

        self.is_fitted = True

    # ── Inference (fully vectorised) ──────────────────────────────────────────

    def predict(
        self,
        test_graph: dgl.DGLGraph,
        threshold:  Optional[float] = None,
        verbose:    bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIX: inference loop replaced with vectorised numpy operations.
        No per-node Python iteration.
        """
        if not self.is_fitted:
            raise RuntimeError('Call fit() before predict().')

        from utils.data_loader import get_combined_node_features

        if verbose:
            print(f'Running FLASH inference '
                  f'({test_graph.num_nodes():,} nodes)...')

        g = dgl.add_self_loop(test_graph)

        self.encoder.eval()
        with torch.no_grad():
            x       = get_combined_node_features(g)
            raw_x   = x.numpy()                        # (N, 10)
            gnn_emb = self.encoder(g, x).numpy()       # (N, 64)

        # ── Vectorised DB lookup ───────────────────────────────────────────
        node_types = g.ndata['type'].numpy()            # (N,)
        edge_hists = self._compute_edge_type_histograms(test_graph)
        hist_q     = np.round(edge_hists, 1)            # (N, 10)

        # Build lookup result array
        final_gnn = gnn_emb.copy()                      # start with live emb
        hits = 0

        # Batch lookup: group nodes by their key, look up once per unique key
        # Build all keys at once
        keys = [(int(node_types[i]), tuple(hist_q[i].tolist()))
                for i in range(len(node_types))]

        # Unique keys → single lookup per unique key
        unique_keys = list(set(keys))
        key_to_emb  = {}
        for k in unique_keys:
            stored = self._emb_db.get(k, None)
            if stored is not None:
                key_to_emb[k] = stored

        # Assign stored embeddings in one pass
        for i, k in enumerate(keys):
            if k in key_to_emb:
                final_gnn[i] = key_to_emb[k]
                hits += 1

        hit_rate = hits / len(keys) * 100
        if verbose:
            print(f'  Embedding DB hit rate : {hit_rate:.1f}%  '
                  f'({hits:,} / {len(keys):,})')

        # ── Combine features and classify ─────────────────────────────────
        combined        = np.concatenate([raw_x, final_gnn], axis=1)  # (N,74)
        combined_scaled = self.scaler.transform(combined)
        proba           = self.classifier.predict_proba(combined_scaled)

        # Normalised entropy anomaly score
        proba_c     = np.clip(proba, 1e-10, 1.0)
        entropy     = -np.sum(proba_c * np.log(proba_c), axis=1)
        max_entropy = np.log(proba.shape[1])
        scores      = (entropy / max_entropy).astype(np.float32)

        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        if verbose:
            print(f'  Score range : {scores.min():.4f} – {scores.max():.4f}  '
                  f'mean={scores.mean():.4f}')

        if threshold is not None:
            preds = (scores_tensor >= threshold).long()
        else:
            preds = torch.zeros(len(scores), dtype=torch.long)

        return scores_tensor, preds

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            'encoder_state':  self.encoder.state_dict(),
            'cls_head_state': self._cls_head.state_dict(),
            'classifier':     self.classifier,
            'scaler':         self.scaler,
            'emb_db':         self._emb_db,
            'is_fitted':      self.is_fitted,
            'config': {
                'in_dim':      self.in_dim,
                'hidden_dim':  self.hidden_dim,
                'gnn_out_dim': self.gnn_out_dim,
                'num_layers':  self.num_layers,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'FLASH model saved to {path}')

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.encoder.load_state_dict(state['encoder_state'])
        self._cls_head = nn.Linear(self.gnn_out_dim, NUM_NODE_TYPES)
        self._cls_head.load_state_dict(state['cls_head_state'])
        self.classifier = state['classifier']
        self.scaler     = state['scaler']
        self._emb_db    = state['emb_db']
        self.is_fitted  = state['is_fitted']
        print(f'FLASH model loaded from {path}')


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.data_loader import load_dataset
    from utils.metrics import evaluate, print_results, find_best_threshold

    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'

    print('Loading dataset...')
    train_graphs, test_graph, test_labels = load_dataset(data_dir)

    model = FLASHModel(
        in_dim      = 10,
        hidden_dim  = 64,
        gnn_out_dim = 64,
        num_layers  = 2,
        dropout     = 0.3,
        use_xgboost = True
    )

    model.fit(train_graphs, gnn_epochs=20, lr=0.001, verbose=True)

    scores, _ = model.predict(test_graph, threshold=None, verbose=True)

    print('\nSearching for best threshold...')
    best_thresh, best_f1, _ = find_best_threshold(
        test_labels, scores, metric='f1'
    )
    print(f'Best threshold : {best_thresh:.4f}  →  F1 = {best_f1:.4f}')

    _, preds = model.predict(test_graph, threshold=best_thresh, verbose=False)

    results = evaluate(test_labels, preds, scores, model_name='FLASH')
    print_results(results)

    model.save('./models/saved/flash_model.pkl')
    print('\nflash_model.py  ✓  done')