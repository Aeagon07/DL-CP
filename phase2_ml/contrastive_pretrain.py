"""
Phase 2 — Contrastive Self-Supervised Pre-training (Module 5 of 12)
=====================================================================
SimCLR-style contrastive learning for operational time-series.
Pre-trains a shared LSTM encoder WITHOUT labels, then fine-tunes
downstream XGBoost/classifier on the learned embeddings.

Why Self-Supervised Learning for breach detection:
  - Breach events are rare (class imbalance problem)
  - SSL pre-training uses ALL data (labeled + unlabeled) to learn
    robust temporal representations
  - Better representations → XGBoost needs fewer labeled breach examples
    to achieve high accuracy

SimCLR Framework Adapted for Time-Series:
  1. Each window x → augmented views x_i, x_j (two different augmentations)
  2. LSTM encoder: f(x) → h  (temporal representation)
  3. Projection head: g(h) → z  (contrastive space)
  4. NT-Xent Loss: pulls augmented views of same window close,
     pushes different windows apart in embedding space

Augmentations (time-series specific):
  - Jitter:        add Gaussian noise to values
  - Time-warp:     stretch/compress time axis slightly
  - Magnitude-scale: multiply values by random scalar
  - Window-slice:  crop a random sub-window and resize

Run standalone:
  python phase2_ml/contrastive_pretrain.py
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
# Time-Series Augmentations
# ─────────────────────────────────────────────────────────────────────────────
class TimeSeriesAugmenter:
    """
    Generates two random views of the same time-series window.
    Each augmentation preserves the overall temporal pattern while
    introducing variation that the model must learn to be invariant to.
    """

    def __init__(
        self,
        jitter_sigma:  float = 0.05,
        scale_range:   tuple = (0.8, 1.2),
        warp_knots:    int   = 4,
        slice_pct:     float = 0.9,
    ):
        self.jitter_sigma = jitter_sigma
        self.scale_range  = scale_range
        self.warp_knots   = warp_knots
        self.slice_pct    = slice_pct

    def jitter(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian noise (augments sensor/measurement noise)."""
        return x + np.random.normal(0, self.jitter_sigma, x.shape).astype(np.float32)

    def magnitude_scale(self, x: np.ndarray) -> np.ndarray:
        """Scale all values by a random factor (augments operational volume shifts)."""
        scale = np.random.uniform(*self.scale_range)
        return (x * scale).astype(np.float32)

    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """
        Warp the time axis by a smooth random curve.
        Simulates irregular sampling intervals.
        """
        seq_len = x.shape[0]
        # Create warp knots and interpolate
        tt  = np.linspace(0, seq_len - 1, seq_len)
        knot_x = np.linspace(0, seq_len - 1, self.warp_knots + 2)
        knot_y = knot_x + np.random.uniform(-seq_len * 0.1, seq_len * 0.1, len(knot_x))
        knot_y = np.clip(knot_y, 0, seq_len - 1)
        knot_y[0]  = 0
        knot_y[-1] = seq_len - 1

        warped_tt = np.interp(tt, knot_x, knot_y)
        warped_tt = np.clip(warped_tt, 0, seq_len - 1)

        # Resample x at warped time points
        result = np.zeros_like(x)
        for f in range(x.shape[1]):
            result[:, f] = np.interp(tt, warped_tt, x[:, f])
        return result.astype(np.float32)

    def window_slice(self, x: np.ndarray) -> np.ndarray:
        """
        Crop a random contiguous sub-window and resize back to original length.
        Augments partial observation windows.
        """
        seq_len = x.shape[0]
        slice_len = max(2, int(seq_len * self.slice_pct))
        start = np.random.randint(0, seq_len - slice_len + 1)
        sliced = x[start:start + slice_len, :]

        # Resize back using linear interpolation
        result = np.zeros_like(x)
        for f in range(x.shape[1]):
            result[:, f] = np.interp(
                np.linspace(0, slice_len - 1, seq_len),
                np.arange(slice_len),
                sliced[:, f],
            )
        return result.astype(np.float32)

    def augment(self, x: np.ndarray) -> np.ndarray:
        """Apply a random combination of 2 augmentations."""
        augmentations = [self.jitter, self.magnitude_scale, self.time_warp, self.window_slice]
        chosen = np.random.choice(len(augmentations), size=2, replace=False)
        view = x.copy()
        for idx in chosen:
            view = augmentations[idx](view)
        return view

    def get_pair(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get two independently augmented views of x."""
        return self.augment(x), self.augment(x)


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Dataset
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveDataset(Dataset):
    """Returns (view_i, view_j) pairs for contrastive training."""

    def __init__(self, X: np.ndarray, augmenter: TimeSeriesAugmenter):
        self.X          = X
        self.augmenter  = augmenter

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        xi, xj = self.augmenter.get_pair(x)
        return torch.tensor(xi, dtype=torch.float32), torch.tensor(xj, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Encoder (LSTM + Projection Head)
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveEncoder(nn.Module):
    """
    LSTM encoder with projection head for contrastive learning.

    Encoder f(·):  LSTM → representation h  (this is what we keep)
    Projection g(·): MLP h → z  (used only during pre-training)
    """

    def __init__(
        self,
        input_size:     int   = 11,
        hidden_size:    int   = 128,
        num_layers:     int   = 2,
        dropout:        float = 0.1,
        proj_dim:       int   = 64,
        repr_dim:       int   = 128,
    ):
        super().__init__()
        # Encoder
        self.encoder = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
            batch_first   = True,
        )
        feat_dim = hidden_size * 2   # bidirectional

        # Projection head (used only during contrastive training)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, repr_dim),
            nn.BatchNorm1d(repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract representation h from input x."""
        _, (h_n, _) = self.encoder(x)     # h_n: (num_layers*2, batch, hidden)
        # Concat forward + backward last-layer hidden states
        h_fwd = h_n[-2]                   # (batch, hidden)
        h_bwd = h_n[-1]                   # (batch, hidden)
        h     = torch.cat([h_fwd, h_bwd], dim=1)  # (batch, hidden*2)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass → normalized projection z."""
        h = self.encode(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)      # unit sphere normalization


# ─────────────────────────────────────────────────────────────────────────────
# NT-Xent Loss (Normalized Temperature-Scaled Cross-Entropy)
# ─────────────────────────────────────────────────────────────────────────────
class NTXentLoss(nn.Module):
    """
    InfoNCE / NT-Xent loss for SimCLR.
    For a batch of N windows, treats the 2N augmented views as:
      - Positive pairs:  (xi_k, xj_k) — two views of the same window k
      - Negative pairs:  all other 2(N-1) views
    Loss = -log( exp(sim(zi,zj)/τ) / Σ_k exp(sim(zi,zk)/τ) )
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: (batch, proj_dim) — normalized projections of view i and j
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)    # (2N, proj_dim)

        # Similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature    # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pair indices: (i, i+N) and (i+N, i)
        pos_i = torch.arange(batch_size, device=z.device)
        pos_j = torch.arange(batch_size, 2 * batch_size, device=z.device)

        # Loss for view i → j and view j → i
        loss_i = -sim[pos_i, pos_j] + torch.logsumexp(sim[pos_i], dim=1)
        loss_j = -sim[pos_j, pos_i] + torch.logsumexp(sim[pos_j], dim=1)

        return (loss_i.mean() + loss_j.mean()) / 2


# ─────────────────────────────────────────────────────────────────────────────
# Contrastive Trainer
# ─────────────────────────────────────────────────────────────────────────────
class ContrastiveTrainer:
    """
    Pre-trains the LSTM encoder using SimCLR NT-Xent loss.
    No labels required — purely self-supervised.
    """

    def __init__(
        self,
        model:       ContrastiveEncoder,
        augmenter:   Optional[TimeSeriesAugmenter] = None,
        temperature: float = 0.5,
        lr:          float = 3e-4,
        epochs:      int   = 50,
        batch_size:  int   = 64,
        model_dir:   str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.augmenter  = augmenter or TimeSeriesAugmenter()
        self.criterion  = NTXentLoss(temperature)
        self.optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def pretrain(
        self,
        X: np.ndarray,          # ALL sequences, labeled OR unlabeled
        queue_name: str = "all",
    ) -> dict:
        """Run SimCLR pre-training. Returns loss history."""
        dataset   = ContrastiveDataset(X, self.augmenter)
        loader    = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

        history   = {"loss": [], "best_epoch": 0}
        best_loss = float("inf")
        best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_losses = []
            for xi, xj in loader:
                xi = xi.to(DEVICE)
                xj = xj.to(DEVICE)
                self.optimizer.zero_grad()
                zi = self.model(xi)
                zj = self.model(xj)
                loss = self.criterion(zi, zj)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = float(np.mean(epoch_losses))
            history["loss"].append(avg_loss)
            logger.info(f"[SSL|{queue_name}] Epoch {epoch:3d} | NT-Xent={avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                history["best_epoch"] = epoch
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state:
            self.model.load_state_dict(best_state)

        ckpt = self.model_dir / f"ssl_encoder_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved SSL encoder → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extractor for Downstream XGBoost
# ─────────────────────────────────────────────────────────────────────────────
class SSLFeatureExtractor:
    """
    Frozen SSL encoder → extract embeddings for XGBoost fine-tuning.
    Concatenates learned representation with original tabular features.
    """

    def __init__(self, encoder: ContrastiveEncoder):
        self.encoder = encoder.to(DEVICE)
        self.encoder.eval()

    def extract(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, seq_len, features)
        Returns: (N, hidden_size*2) — LSTM representations
        """
        self.encoder.eval()
        all_repr = []
        with torch.no_grad():
            for i in range(0, len(X), 64):
                batch = torch.tensor(X[i:i+64], dtype=torch.float32).to(DEVICE)
                h     = self.encoder.encode(batch)       # (batch, hidden*2)
                all_repr.append(h.cpu().numpy())
        return np.concatenate(all_repr, axis=0)

    @classmethod
    def from_checkpoint(cls, path: str, input_size: int = 11) -> "SSLFeatureExtractor":
        model = ContrastiveEncoder(input_size=input_size)
        ckpt  = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded SSL encoder from {path}")
        return cls(model)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration
# ─────────────────────────────────────────────────────────────────────────────
def log_ssl_to_mlflow(
    queue_name: str,
    history: dict,
    n_unlabeled: int,
    tracking_uri: Optional[str] = None,
):
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("contrastive_ssl")
        with mlflow.start_run(run_name=f"ssl_{queue_name.replace(' ', '_')}"):
            mlflow.log_param("queue_name",   queue_name)
            mlflow.log_param("n_unlabeled",  n_unlabeled)
            mlflow.log_param("best_epoch",   history.get("best_epoch", 0))
            if history["loss"]:
                mlflow.log_metric("final_ntxent_loss", history["loss"][-1])
                mlflow.log_metric("best_ntxent_loss",  min(history["loss"]))
        logger.info(f"SSL metrics logged for {queue_name}")
    except Exception as e:
        logger.warning(f"MLflow skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 5 — Contrastive SSL Pre-training Smoke Test")
    print("=" * 60)

    rng  = np.random.default_rng(42)
    N, T, F = 200, 12, 11
    X    = rng.normal(0, 1, (N, T, F)).astype(np.float32)

    aug    = TimeSeriesAugmenter()
    xi, xj = aug.get_pair(X[0])
    print(f"\n  Original shape:    {X[0].shape}")
    print(f"  View i shape:      {xi.shape}")
    print(f"  View j shape:      {xj.shape}")
    print(f"  Views differ:      {not np.allclose(xi, xj)}")

    model   = ContrastiveEncoder(input_size=F)
    trainer = ContrastiveTrainer(model, aug, epochs=3, batch_size=32)
    history = trainer.pretrain(X, queue_name="test")
    print(f"\n  Final NT-Xent loss: {history['loss'][-1]:.4f}")

    extractor = SSLFeatureExtractor(model)
    embeddings = extractor.extract(X)
    print(f"  Embedding shape:    {embeddings.shape}")   # (N, hidden*2=256)
    print(f"  Embedding mean:     {embeddings.mean():.4f}")
    print("\n✓ Module 5 (Contrastive SSL) — PASSED\n")
