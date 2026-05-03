"""
Phase 2 — Graph Neural Network for Queue Dependencies (Module 9 of 12)
========================================================================
Models the 6 Appian queues as a directed graph where edges represent
business-process handoff relationships. A Graph Attention Network (GAT)
learns cross-queue dependency weights dynamically from data.

Key Insight:
  Tabular models treat queues independently. But in reality:
  "Payment Processing overload → Risk Assessment will overflow in 2h"
  "Document Review backlog → Audit Preparation will breach SLA in 4h"

  GNN captures these cascade effects that LSTM/XGBoost completely miss.

Graph Structure:
  Nodes: 6 queues (node features = current feature vector per queue)
  Edges: directed handoff relationships between queues
  Edge weights: learned attention coefficients α_ij

  Adjacency (based on Appian BPM process flow):
    Payment Processing  → Risk Assessment
    Risk Assessment     → Compliance Check
    Document Review     → Audit Preparation
    Customer Onboarding → Compliance Check
    Compliance Check    → Audit Preparation
    Audit Preparation   → Compliance Check  (feedback cycle)

Graph Attention Network (GAT):
  h'_i = σ( Σ_j∈N(i) α_ij · W · h_j )
  α_ij = softmax( LeakyReLU( a^T [W·h_i || W·h_j] ) )

Implemented in PURE PyTorch (no PyTorch Geometric dependency).

Run standalone:
  python phase2_ml/gnn_dependency.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Queue Graph Definition
# ─────────────────────────────────────────────────────────────────────────────
QUEUE_NAMES = [
    "Document Review",      # 0
    "Compliance Check",     # 1
    "Payment Processing",   # 2
    "Customer Onboarding",  # 3
    "Risk Assessment",       # 4
    "Audit Preparation",    # 5
]

# Directed edges: (source_idx, target_idx) based on Appian BPM process flow
QUEUE_EDGES = [
    (2, 4),   # Payment Processing → Risk Assessment
    (4, 1),   # Risk Assessment    → Compliance Check
    (0, 5),   # Document Review    → Audit Preparation
    (3, 1),   # Customer Onboarding→ Compliance Check
    (1, 5),   # Compliance Check   → Audit Preparation
    (5, 1),   # Audit Preparation  → Compliance Check (feedback)
    (0, 1),   # Document Review    → Compliance Check
    (2, 1),   # Payment Processing → Compliance Check
]


class QueueGraph:
    """
    Manages the queue dependency graph structure.
    Precomputes sparse edge tensors for GAT layers.
    """

    def __init__(
        self,
        num_nodes: int = 6,
        edges: Optional[list[tuple[int, int]]] = None,
    ):
        self.num_nodes = num_nodes
        self.edges     = edges or QUEUE_EDGES
        self.num_edges = len(self.edges)

        # Edge tensors (src, dst)
        src = [e[0] for e in self.edges]
        dst = [e[1] for e in self.edges]
        self.edge_src = torch.tensor(src, dtype=torch.long)
        self.edge_dst = torch.tensor(dst, dtype=torch.long)

        # Adjacency matrix (for visualization)
        self.adj = torch.zeros(num_nodes, num_nodes)
        for s, d in self.edges:
            self.adj[s, d] = 1.0

    def to(self, device: torch.device) -> "QueueGraph":
        self.edge_src = self.edge_src.to(device)
        self.edge_dst = self.edge_dst.to(device)
        self.adj      = self.adj.to(device)
        return self

    def describe(self) -> str:
        lines = ["Queue Dependency Graph:"]
        for s, d in self.edges:
            lines.append(f"  {QUEUE_NAMES[s]} → {QUEUE_NAMES[d]}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Layer (GAT)
# ─────────────────────────────────────────────────────────────────────────────
class GraphAttentionLayer(nn.Module):
    """
    Single Graph Attention Layer.

    For each node i:
      1. Linear transform: z_i = W · h_i
      2. Compute attention coefficient for each neighbor j:
           e_ij = LeakyReLU( a^T [z_i || z_j] )
      3. Normalize: α_ij = softmax_j(e_ij)
      4. Aggregate: h'_i = σ( Σ_j α_ij · z_j )
    """

    def __init__(
        self,
        in_features:   int,
        out_features:  int,
        dropout:       float = 0.1,
        alpha:         float = 0.2,
        concat:        bool  = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.dropout      = dropout
        self.concat       = concat

        self.W       = nn.Parameter(torch.empty(in_features, out_features))
        self.a       = nn.Parameter(torch.empty(2 * out_features))
        self.leaky   = nn.LeakyReLU(alpha)
        self.drop    = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.unsqueeze(0))
        nn.init.xavier_uniform_(self.a.view(1, -1))

    def forward(
        self,
        h:         torch.Tensor,      # (batch, num_nodes, in_features)
        graph:     QueueGraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        h: (batch, N, in_features)
        Returns: (h_new, attn_matrix)
          h_new:       (batch, N, out_features)
          attn_matrix: (batch, N, N) — attention weights (interpretable)
        """
        B, N, _ = h.shape

        # Linear projection
        z = h @ self.W                          # (B, N, out_features)

        # ── Attention coefficients ─────────────────────────────────
        src = graph.edge_src  # (num_edges,)
        dst = graph.edge_dst  # (num_edges,)

        z_src = z[:, src, :]   # (B, E, out_features)
        z_dst = z[:, dst, :]   # (B, E, out_features)

        edge_feat = torch.cat([z_src, z_dst], dim=-1)   # (B, E, 2*out_features)
        e         = self.leaky(edge_feat @ self.a)       # (B, E)

        # ── Sparse softmax per destination node ────────────────────
        # Build (B, N, N) attention matrix — only populated at edge positions
        attn_matrix = torch.full((B, N, N), float("-inf"), device=h.device)
        attn_matrix[:, src, dst] = e

        attn_matrix = F.softmax(attn_matrix, dim=-1)    # (B, N, N)
        attn_matrix = torch.nan_to_num(attn_matrix, nan=0.0)  # handle isolated nodes

        attn_dropped = self.drop(attn_matrix)

        # ── Aggregate ─────────────────────────────────────────────
        h_new = attn_dropped @ z                        # (B, N, out_features)

        if self.concat:
            h_new = F.elu(h_new)

        return h_new, attn_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Head GAT
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention: K parallel attention heads, outputs concatenated."""

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        num_heads:    int   = 4,
        dropout:      float = 0.1,
        concat:       bool  = True,
    ):
        super().__init__()
        self.concat   = concat
        self.num_heads = num_heads
        head_out = out_features // num_heads if concat else out_features

        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, head_out, dropout, concat=concat)
            for _ in range(num_heads)
        ])

    def forward(
        self, h: torch.Tensor, graph: QueueGraph
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (aggregated_features, mean_attention_matrix)
        """
        head_outputs = [head(h, graph) for head in self.heads]
        h_list  = [o[0] for o in head_outputs]
        a_list  = [o[1] for o in head_outputs]

        if self.concat:
            h_out = torch.cat(h_list, dim=-1)       # (B, N, out_features)
        else:
            h_out = torch.stack(h_list).mean(dim=0)

        mean_attn = torch.stack(a_list).mean(dim=0)  # (B, N, N)
        return h_out, mean_attn


# ─────────────────────────────────────────────────────────────────────────────
# GNN Dependency Model
# ─────────────────────────────────────────────────────────────────────────────
class GNNDependencyModel(nn.Module):
    """
    2-layer Graph Attention Network for cross-queue breach risk prediction.

    Input:  per-queue feature snapshots for all 6 queues simultaneously
    Output: per-queue breach risk scores incorporating cross-queue signals
    """

    def __init__(
        self,
        node_features: int   = 11,
        hidden_dim:    int   = 64,
        output_dim:    int   = 32,
        num_heads:     int   = 4,
        dropout:       float = 0.1,
    ):
        super().__init__()

        # Layer 1: GAT with multi-head
        self.gat1 = MultiHeadGAT(node_features, hidden_dim, num_heads, dropout, concat=True)

        # Layer 2: GAT averaging over heads
        self.gat2 = MultiHeadGAT(hidden_dim, output_dim, num_heads=1, dropout=dropout, concat=False)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

        # Per-node breach risk head
        self.risk_head = nn.Sequential(
            nn.Linear(output_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x:     torch.Tensor,     # (batch, num_nodes, node_features)
        graph: QueueGraph,
    ) -> dict:
        """
        Returns: node_risk_scores, layer2_attn_matrix
          node_risk_scores: (batch, num_nodes) — breach probability per queue
          attn_weights:     (batch, num_nodes, num_nodes) — interpretable!
        """
        # Layer 1
        h1, _ = self.gat1(x, graph)
        h1     = self.layer_norm1(h1)

        # Layer 2
        h2, attn2 = self.gat2(h1, graph)
        h2         = self.layer_norm2(h2)

        # Per-node risk
        risk_scores = self.risk_head(h2).squeeze(-1)   # (batch, num_nodes)

        return {
            "risk_scores":  risk_scores,
            "attn_weights": attn2,
            "node_repr":    h2,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (multi-queue snapshots)
# ─────────────────────────────────────────────────────────────────────────────
class GNNDataset(Dataset):
    """
    Each sample: feature matrix for ALL queues at one snapshot_time.
    X: (num_nodes, node_features)
    y: (num_nodes,) — breach labels per queue
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, num_nodes, features), y: (N, num_nodes)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class GNNTrainer:
    def __init__(
        self,
        model:      GNNDependencyModel,
        graph:      QueueGraph,
        lr:         float = 1e-3,
        epochs:     int   = 50,
        batch_size: int   = 32,
        patience:   int   = 10,
        model_dir:  str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.graph      = graph.to(DEVICE)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.criterion  = nn.BCELoss()
        self.optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> dict:
        tr_loader  = DataLoader(GNNDataset(X_tr, y_tr),   self.batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(GNNDataset(X_val, y_val), self.batch_size, shuffle=False, num_workers=0)

        history  = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val = float("inf"); patience_cnt = 0; best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tr_losses = []
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                out  = self.model(Xb, self.graph)
                loss = self.criterion(out["risk_scores"], yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    out = self.model(Xb, self.graph)
                    val_losses.append(self.criterion(out["risk_scores"], yb).item())

            t_l = float(np.mean(tr_losses))
            v_l = float(np.mean(val_losses))
            history["train_loss"].append(t_l)
            history["val_loss"].append(v_l)
            logger.info(f"[GNN] Epoch {epoch:3d} | train={t_l:.4f} val={v_l:.4f}")

            if v_l < best_val:
                best_val = v_l; history["best_epoch"] = epoch; patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience: break

        if best_state: self.model.load_state_dict(best_state)
        ckpt = self.model_dir / "gnn_queue_dependency.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved GNN → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
class GNNInference:
    """GNN inference returning cross-queue risk scores + attention weights."""

    def __init__(self, model: GNNDependencyModel, graph: QueueGraph):
        self.model = model.to(DEVICE)
        self.graph = graph.to(DEVICE)
        self.model.eval()

    def predict(self, X: np.ndarray) -> dict:
        """
        X: (num_nodes, features) or (1, num_nodes, features_)
        Returns per-queue risk scores + attention matrix
        """
        if X.ndim == 2: X = X[np.newaxis, ...]
        xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = self.model(xt, self.graph)

        risk   = out["risk_scores"][0].cpu().numpy()
        attn   = out["attn_weights"][0].cpu().numpy()

        queue_risk = {QUEUE_NAMES[i]: round(float(risk[i]), 4) for i in range(len(QUEUE_NAMES))}

        # Strongest dependency edges
        edge_attns = []
        for s, d in QUEUE_EDGES:
            edge_attns.append({
                "from": QUEUE_NAMES[s],
                "to":   QUEUE_NAMES[d],
                "attention": round(float(attn[s, d]), 4),
            })
        edge_attns.sort(key=lambda x: -x["attention"])

        return {
            "queue_risk_scores": queue_risk,
            "top_dependencies":  edge_attns[:3],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 9 — GNN Queue Dependency Smoke Test")
    print("=" * 60)

    graph = QueueGraph()
    print(f"\n{graph.describe()}\n")

    rng     = np.random.default_rng(42)
    N       = 200
    n_nodes = 6
    F       = 11
    X       = rng.normal(0, 1, (N, n_nodes, F)).astype(np.float32)
    y       = rng.integers(0, 2, (N, n_nodes)).astype(np.float32)
    split   = int(N * 0.8)

    model   = GNNDependencyModel(node_features=F, hidden_dim=32, output_dim=16, num_heads=2)
    trainer = GNNTrainer(model, graph, epochs=3, batch_size=16)
    history = trainer.fit(X[:split], y[:split], X[split:], y[split:])

    infer  = GNNInference(model, graph)
    result = infer.predict(X[0])
    print("  Queue risk scores:")
    for q, r in result["queue_risk_scores"].items():
        print(f"    {q}: {r:.4f}")
    print("  Top dependencies:")
    for dep in result["top_dependencies"]:
        print(f"    {dep['from']} → {dep['to']}: α={dep['attention']:.4f}")
    print(f"\n  Final val loss: {history['val_loss'][-1]:.4f}")
    print("\n✓ Module 9 (GNN Queue Dependency) — PASSED\n")
