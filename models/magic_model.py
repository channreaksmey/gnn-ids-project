"""
models/magic_model.py
=====================
MAGIC: Masked Graph Representation Learning for APT Detection
Based on: Jia et al., USENIX Security 2024

Architecture overview:
  ┌─────────────────────────────────────────────────────┐
  │  Training (benign graphs only)                      │
  │                                                     │
  │  Graph → Mask 50% nodes → GAT Encoder               │
  │       → Node embeddings → GAT Decoder               │
  │       → Reconstruct masked features → Loss          │
  │                                                     │
  │  After training: store all benign node embeddings   │
  │  in a BallTree for fast KNN lookup                  │
  └─────────────────────────────────────────────────────┘
  ┌─────────────────────────────────────────────────────┐
  │  Inference (test graph)                             │
  │                                                     │
  │  Graph → GAT Encoder (no masking)                   │
  │       → Node embeddings                             │
  │       → KNN search in benign embedding store        │
  │       → Anomaly score = avg distance to k neighbours│
  │       → Threshold → binary predictions              │
  └─────────────────────────────────────────────────────┘

Usage:
    from models.magic_model import MAGICModel

    model = MAGICModel(in_dim=10, hidden_dim=64, num_layers=3)
    model.fit(train_graphs)
    scores, preds = model.predict(test_graph)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
import numpy as np
from sklearn.neighbors import BallTree
from typing import List, Tuple, Optional
import pickle


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_NODE_TYPES = 5
NUM_EDGE_TYPES = 10


# ── GAT Encoder ───────────────────────────────────────────────────────────────

class GATEncoder(nn.Module):
    """
    Multi-layer Graph Attention Network encoder.

    Each layer aggregates information from a node's neighbours using
    learned attention weights — neighbours that are more relevant to
    reconstructing this node's features get higher weights.

    The final embedding concatenates the original features with the
    output of every GAT layer, so deeper layers capture wider
    neighbourhood context (1-hop, 2-hop, 3-hop).
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        num_layers: int = 3,
        num_heads:  int = 1,
        dropout:    float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout    = dropout

        self.convs = nn.ModuleList()
        # First layer: in_dim → hidden_dim
        self.convs.append(
            GATConv(in_dim, hidden_dim, num_heads=num_heads,
                    feat_drop=dropout, attn_drop=dropout,
                    activation=F.elu)
        )
        # Subsequent layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim,
                        num_heads=num_heads,
                        feat_drop=dropout, attn_drop=dropout,
                        activation=F.elu)
            )

        self.num_heads  = num_heads
        self.hidden_dim = hidden_dim

    def forward(
        self,
        graph: dgl.DGLGraph,
        x:     torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        graph : DGLGraph
        x     : (num_nodes, in_dim) — node features (possibly masked)

        Returns
        -------
        h : (num_nodes, in_dim + hidden_dim * num_layers)
            Concatenation of input + all layer outputs (MAGIC design).
        """
        layer_outputs = [x]
        h = x

        for conv in self.convs:
            h = conv(graph, h)                         # (N, heads, hidden)
            h = h.mean(dim=1)                          # (N, hidden) avg heads
            layer_outputs.append(h)

        # Concatenate original features with all layer outputs
        return torch.cat(layer_outputs, dim=-1)


# ── GAT Decoder ───────────────────────────────────────────────────────────────

class GATDecoder(nn.Module):
    """
    Single GAT layer that reconstructs the original node features
    from the encoder's output embeddings.

    This provides the training signal: if the encoder produced good
    embeddings, the decoder should be able to recover the features
    that were masked during training.
    """

    def __init__(self, embed_dim: int, out_dim: int, num_heads: int = 1):
        super().__init__()
        self.conv = GATConv(embed_dim, out_dim, num_heads=num_heads,
                            activation=None)

    def forward(
        self,
        graph: dgl.DGLGraph,
        h:     torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (num_nodes, embed_dim) — encoder output embeddings

        Returns
        -------
        recon : (num_nodes, out_dim) — reconstructed node features
        """
        out = self.conv(graph, h)    # (N, heads, out_dim)
        return out.mean(dim=1)       # (N, out_dim)


# ── Full MAGIC Model ──────────────────────────────────────────────────────────

class MAGICModel(nn.Module):
    """
    Complete MAGIC model: encoder + decoder + masking + KNN detector.

    Call fit() to train on benign graphs, then predict() on the test graph.
    """

    def __init__(
        self,
        in_dim:     int   = 10,    # use get_combined_node_features → 10 dims
        hidden_dim: int   = 64,
        num_layers: int   = 3,
        num_heads:  int   = 1,
        mask_rate:  float = 0.5,
        k_neighbors: int  = 10,
        dropout:    float = 0.1
    ):
        """
        Parameters
        ----------
        in_dim       : input feature dimension (10 if using combined features)
        hidden_dim   : hidden embedding dimension per GAT layer
        num_layers   : number of GAT layers (depth of neighbourhood context)
        num_heads    : number of attention heads per GAT layer
        mask_rate    : fraction of nodes to mask during training (paper uses 0.5)
        k_neighbors  : k for KNN anomaly detection (paper uses 10)
        dropout      : dropout rate in GAT layers
        """
        super().__init__()

        self.in_dim      = in_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.mask_rate   = mask_rate
        self.k_neighbors = k_neighbors

        # Embedding dimension after concatenating all layer outputs:
        # original features (in_dim) + hidden_dim per layer
        self.embed_dim = in_dim + hidden_dim * num_layers

        # Learnable mask token — replaces masked node features
        # Initialised to zero; the model learns what value works best
        self.mask_token = nn.Parameter(torch.zeros(1, in_dim))

        # Re-masking projection (applied before decoder, paper Sec 4.2.3)
        self.remasking_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Encoder and decoder
        self.encoder = GATEncoder(in_dim, hidden_dim, num_layers,
                                   num_heads, dropout)
        self.decoder = GATDecoder(self.embed_dim, in_dim, num_heads)

        # KNN store — filled after training, used during inference
        self._ball_tree:      Optional[BallTree] = None
        self._train_emb_mean: Optional[float]    = None
        self._train_emb_std:  Optional[float]    = None
        self.is_fitted = False

    # ── Masking ───────────────────────────────────────────────────────────────

    def _mask_nodes(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly replace mask_rate fraction of node features with
        the learnable mask token.

        Returns
        -------
        x_masked      : (N, in_dim) — features with masked nodes replaced
        masked_idx    : indices of masked nodes
        unmasked_idx  : indices of unmasked nodes
        """
        N = x.size(0)
        num_masked = max(1, int(N * self.mask_rate))

        perm          = torch.randperm(N, device=x.device)
        masked_idx    = perm[:num_masked]
        unmasked_idx  = perm[num_masked:]

        x_masked = x.clone()
        x_masked[masked_idx] = self.mask_token.expand(num_masked, -1)

        return x_masked, masked_idx, unmasked_idx

    # ── Reconstruction loss ────────────────────────────────────────────────────

    def _scaled_cosine_loss(
        self,
        original:      torch.Tensor,
        reconstructed: torch.Tensor,
        gamma:         int = 3
    ) -> torch.Tensor:
        """
        Scaled cosine error loss (Equation from MAGIC paper, Sec 4.2.3).

        loss = mean( (1 - cosine_similarity)^gamma )

        gamma=3 makes the loss scale sharply — easy samples (high similarity)
        contribute very little loss while hard samples drive training.
        """
        cos_sim = F.cosine_similarity(original, reconstructed, dim=-1)
        loss = (1.0 - cos_sim).pow(gamma).mean()
        return loss

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        graph: dgl.DGLGraph,
        x:     torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        graph    : DGLGraph
        x        : (N, in_dim) node features
        training : if True, apply masking and return reconstruction loss

        Returns
        -------
        node_embeddings : (N, embed_dim)
        loss            : scalar tensor if training=True, else None
        """
        if training:
            # ── Mask → encode → re-mask → decode → loss ───────────────────
            x_masked, masked_idx, _ = self._mask_nodes(x)

            # Encode with masked features
            node_embeddings = self.encoder(graph, x_masked)

            # Re-masking before decoder (MAGIC paper Sec 4.2.3):
            # replace masked nodes with a re-mask token in embedding space
            h_star = self.remasking_proj(node_embeddings)
            remask_val = torch.zeros_like(h_star[0:1])
            h_star[masked_idx] = remask_val.expand(len(masked_idx), -1)

            # Decode to reconstruct original features
            recon = self.decoder(graph, h_star)

            # Loss only on masked nodes
            loss = self._scaled_cosine_loss(
                x[masked_idx], recon[masked_idx]
            )
            return node_embeddings, loss

        else:
            # ── Inference: no masking ──────────────────────────────────────
            node_embeddings = self.encoder(graph, x)
            return node_embeddings, None

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_graphs: List[dgl.DGLGraph],
        epochs:       int   = 30,
        lr:           float = 0.001,
        weight_decay: float = 5e-4,
        batch_graphs: int   = 1,
        verbose:      bool  = True
    ) -> List[float]:
        """
        Train the masked autoencoder on benign graphs, then build the
        KNN index from the learned embeddings.

        Parameters
        ----------
        train_graphs  : list of DGLGraph (benign only)
        epochs        : training epochs
        lr            : learning rate (paper uses 0.001)
        weight_decay  : L2 regularisation (paper uses 5e-4)
        batch_graphs  : how many graphs to process per gradient step
                        (use 1 on Mac to stay within memory)
        verbose       : print loss per epoch

        Returns
        -------
        loss_history : list of average loss per epoch
        """
        from utils.data_loader import get_combined_node_features

        optimizer    = torch.optim.Adam(self.parameters(),
                                         lr=lr, weight_decay=weight_decay)
        loss_history = []

        if verbose:
            print(f'Training MAGIC encoder...')
            print(f'  Graphs     : {len(train_graphs)}')
            print(f'  Epochs     : {epochs}')
            print(f'  in_dim     : {self.in_dim}')
            print(f'  hidden_dim : {self.hidden_dim}')
            print(f'  num_layers : {self.num_layers}')
            print(f'  mask_rate  : {self.mask_rate}')

        self.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches  = 0

            for graph in train_graphs:
                # Add self-loops so every node has at least one incoming edge.
                # GAT requires this — nodes with zero in-degree produce
                # invalid attention outputs.
                graph = dgl.add_self_loop(graph)
                x = get_combined_node_features(graph)  # (N, 10)

                optimizer.zero_grad()
                _, loss = self.forward(graph, x, training=True)
                loss.backward()
                # Gradient clipping — prevents exploding gradients on
                # large graphs
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            loss_history.append(avg_loss)

            if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
                print(f'  Epoch {epoch+1:>3}/{epochs}  loss={avg_loss:.6f}')

        # ── Build KNN index from benign training embeddings ────────────────
        if verbose:
            print('\nBuilding KNN index from benign embeddings...')

        self._build_knn_index(train_graphs, verbose=verbose)
        self.is_fitted = True

        if verbose:
            print('MAGIC training complete  ✓')

        return loss_history

    def _build_knn_index(
        self,
        train_graphs: List[dgl.DGLGraph],
        verbose:      bool = True
    ) -> None:
        """
        Run all training nodes through the trained encoder (no masking)
        and store embeddings in a BallTree for fast KNN search.

        BallTree is much faster than a naive distance loop for 1M+ nodes.
        """
        from utils.data_loader import get_combined_node_features

        self.eval()
        all_embeddings = []

        with torch.no_grad():
            for i, graph in enumerate(train_graphs):
                graph = dgl.add_self_loop(graph)
                x   = get_combined_node_features(graph)
                emb, _ = self.forward(graph, x, training=False)
                all_embeddings.append(emb.cpu().numpy())
                if verbose:
                    print(f'  Encoded train graph {i}:  '
                          f'{emb.shape[0]:,} embeddings  '
                          f'dim={emb.shape[1]}')

        all_emb = np.vstack(all_embeddings)   # (total_train_nodes, embed_dim)

        # Normalise embeddings before KNN (improves distance comparisons)
        self._train_emb_mean = all_emb.mean(axis=0)
        self._train_emb_std  = all_emb.std(axis=0) + 1e-8
        all_emb_norm = (all_emb - self._train_emb_mean) / self._train_emb_std

        if verbose:
            print(f'  Total benign embeddings stored : {len(all_emb_norm):,}')
            print(f'  Embedding dimension            : {all_emb_norm.shape[1]}')
            print(f'  Building BallTree...')

        # BallTree with Euclidean metric
        # leaf_size=40 is a good default — larger = less memory, slower query
        self._ball_tree = BallTree(all_emb_norm, leaf_size=40, metric='euclidean')

        # Compute average KNN distance among training nodes
        # (used to normalise anomaly scores at inference time)
        sample_size = min(5000, len(all_emb_norm))
        sample_idx  = np.random.choice(len(all_emb_norm),
                                        sample_size, replace=False)
        sample      = all_emb_norm[sample_idx]
        dists, _    = self._ball_tree.query(sample, k=self.k_neighbors + 1)
        # Skip column 0 (distance to self = 0)
        self._avg_train_dist = dists[:, 1:].mean()

        if verbose:
            print(f'  Avg KNN distance (train)       : '
                  f'{self._avg_train_dist:.6f}')

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        test_graph: dgl.DGLGraph,
        threshold:  Optional[float] = None,
        verbose:    bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Produce anomaly scores and binary predictions for every node
        in the test graph.

        Anomaly score = avg KNN distance to benign embeddings,
                        normalised by avg training KNN distance.

        Parameters
        ----------
        test_graph : DGLGraph
        threshold  : float or None.
                     If None, returns only scores.
                     Use find_best_threshold() from metrics.py to choose.

        Returns
        -------
        scores : FloatTensor (num_nodes,)  — higher = more anomalous
        preds  : LongTensor  (num_nodes,)  — 0=benign, 1=malicious
        """
        if not self.is_fitted:
            raise RuntimeError('Call fit() before predict().')

        from utils.data_loader import get_combined_node_features

        if verbose:
            print(f'Running MAGIC inference on test graph '
                  f'({test_graph.num_nodes():,} nodes)...')

        self.eval()
        with torch.no_grad():
            test_graph = dgl.add_self_loop(test_graph)
            x   = get_combined_node_features(test_graph)
            emb, _ = self.forward(test_graph, x, training=False)
            emb_np = emb.cpu().numpy()

        # Normalise using training statistics
        emb_norm = (emb_np - self._train_emb_mean) / self._train_emb_std

        # KNN query — returns distances to k nearest benign neighbours
        if verbose:
            print(f'  Querying BallTree '
                  f'(k={self.k_neighbors})...')

        dists, _ = self._ball_tree.query(emb_norm, k=self.k_neighbors)
        avg_dists = dists.mean(axis=1)                      # (N,)

        # Normalise by average training distance
        # score > 1.0 means farther from benign than average
        scores = avg_dists / (self._avg_train_dist + 1e-8)  # (N,)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)

        if verbose:
            print(f'  Score range: '
                  f'{scores.min():.4f} – {scores.max():.4f}  '
                  f'mean={scores.mean():.4f}')

        if threshold is not None:
            preds = (scores_tensor >= threshold).long()
        else:
            preds = torch.zeros(len(scores), dtype=torch.long)

        return scores_tensor, preds

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save model weights and KNN index to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        state = {
            'model_state':      self.state_dict(),
            'ball_tree':        self._ball_tree,
            'train_emb_mean':   self._train_emb_mean,
            'train_emb_std':    self._train_emb_std,
            'avg_train_dist':   self._avg_train_dist,
            'is_fitted':        self.is_fitted,
            'config': {
                'in_dim':      self.in_dim,
                'hidden_dim':  self.hidden_dim,
                'num_layers':  self.num_layers,
                'mask_rate':   self.mask_rate,
                'k_neighbors': self.k_neighbors,
            }
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f'MAGIC model saved to {path}')

    def load(self, path: str) -> None:
        """Load model weights and KNN index from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.load_state_dict(state['model_state'])
        self._ball_tree       = state['ball_tree']
        self._train_emb_mean  = state['train_emb_mean']
        self._train_emb_std   = state['train_emb_std']
        self._avg_train_dist  = state['avg_train_dist']
        self.is_fitted        = state['is_fitted']
        print(f'MAGIC model loaded from {path}')


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.data_loader import load_dataset, print_dataset_summary
    from utils.metrics import evaluate, print_results, find_best_threshold

    data_dir = sys.argv[1] if len(sys.argv) > 1 else './data'

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading dataset...')
    train_graphs, test_graph, test_labels = load_dataset(data_dir)

    # ── Train ─────────────────────────────────────────────────────────────────
    model = MAGICModel(
        in_dim     = 10,    # combined features (5 attr + 5 type onehot)
        hidden_dim = 64,
        num_layers = 3,
        mask_rate  = 0.5,
        k_neighbors= 10,
        dropout    = 0.1
    )

    loss_history = model.fit(
        train_graphs,
        epochs  = 30,
        lr      = 0.001,
        verbose = True
    )

    print(f'\nFinal training loss: {loss_history[-1]:.6f}')

    # ── Predict ───────────────────────────────────────────────────────────────
    scores, _ = model.predict(test_graph, threshold=None, verbose=True)

    # Find best threshold
    print('\nSearching for best threshold...')
    best_thresh, best_f1, _ = find_best_threshold(
        test_labels, scores, metric='f1'
    )
    print(f'Best threshold : {best_thresh:.4f}  →  F1 = {best_f1:.4f}')

    _, preds = model.predict(test_graph, threshold=best_thresh, verbose=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = evaluate(test_labels, preds, scores, model_name='MAGIC')
    print_results(results)

    # ── Save ──────────────────────────────────────────────────────────────────
    model.save('./models/saved/magic_model.pkl')

    print('\nmagic_model.py  ✓  done')