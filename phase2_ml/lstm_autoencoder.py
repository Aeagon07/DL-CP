"""
Phase 2 — LSTM Autoencoder Anomaly Detector (Module 2 of 12)
==============================================================
Unsupervised anomaly detection via Encoder-Decoder LSTM.

Strategy:
  1. Train ONLY on normal operational windows (no breach events)
  2. Encoder compresses sequence → bottleneck representation
  3. Decoder reconstructs the original sequence
  4. Reconstruction error (MSE) = anomaly score
  5. Threshold = 95th percentile of validation reconstruction errors

Why it works:
  The model learns the manifold of normal operations. Anomalous events
  (sudden WIP spikes, agent shortages, SLA cascades) cannot be reconstructed
  accurately, producing high reconstruction error → anomaly alert.

Architecture:
  Input  (batch, seq_len, features)
    → Encoder LSTM [2 layers, hidden=64]  → context vector (batch, 64)
    → Repeat context  →  (batch, seq_len, 64)
    → Decoder LSTM [2 layers, hidden=64]
    → Linear projection  →  (batch, seq_len, features)
  Loss: MSE(input, reconstructed)

Run standalone:
  python phase2_ml/lstm_autoencoder.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    """Sliding-window dataset for autoencoder training."""

    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class LSTMAutoencoder(nn.Module):
    """
    Encoder-Decoder LSTM for sequence-to-sequence reconstruction.

    Architecture:
      Encoder: LSTM → last hidden state = context vector
      Decoder: context repeated seq_len times → LSTM → linear → reconstruction
    """

    def __init__(
        self,
        input_size:  int = 11,
        hidden_size: int = 64,
        latent_size: int = 16,
        num_layers:  int = 2,
        dropout:     float = 0.1,
        seq_len:     int = 12,
    ):
        super().__init__()
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # Encoder
        self.encoder = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.encoder_to_latent = nn.Linear(hidden_size, latent_size)

        # Decoder
        self.latent_to_decoder = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.LSTM(
            input_size  = hidden_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.output_proj = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to latent vector. x: (batch, seq_len, features)"""
        _, (h_n, _) = self.encoder(x)         # h_n: (num_layers, batch, hidden)
        context = h_n[-1]                       # last layer: (batch, hidden)
        latent  = self.encoder_to_latent(context)  # (batch, latent_size)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to sequence."""
        h0 = self.latent_to_decoder(latent)    # (batch, hidden)
        h0 = h0.unsqueeze(0).repeat(           # (num_layers, batch, hidden)
            self.decoder.num_layers, 1, 1
        )
        c0 = torch.zeros_like(h0)

        # Use latent as repeated input
        dec_input = h0[-1].unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden)
        out, _ = self.decoder(dec_input, (h0, c0))  # (batch, seq_len, hidden)
        recon  = self.output_proj(out)               # (batch, seq_len, features)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.decode(latent)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer (trains ONLY on normal data)
# ─────────────────────────────────────────────────────────────────────────────
class AutoencoderTrainer:
    """
    Trains the LSTM Autoencoder on normal (non-breach) operational windows.
    Calibrates anomaly threshold on a validation split.
    """

    def __init__(
        self,
        model:      LSTMAutoencoder,
        lr:         float = 1e-3,
        epochs:     int   = 40,
        batch_size: int   = 32,
        patience:   int   = 8,
        model_dir:  str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(
        self,
        X_normal: np.ndarray,        # (N, seq_len, features) — NO breach windows
        queue_name: str = "default",
    ) -> dict:
        """Train autoencoder on normal sequences only."""
        split = int(len(X_normal) * 0.85)
        X_tr, X_val = X_normal[:split], X_normal[split:]

        tr_ds  = SequenceDataset(X_tr)
        val_ds = SequenceDataset(X_val)
        tr_loader  = DataLoader(tr_ds,  batch_size=self.batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        history       = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val      = float("inf")
        patience_cnt  = 0
        best_state    = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────
            self.model.train()
            tr_losses = []
            for Xb in tr_loader:
                Xb = Xb.to(DEVICE)
                self.optimizer.zero_grad()
                recon = self.model(Xb)
                loss  = self.criterion(recon, Xb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_losses.append(loss.item())

            # ── Validate ──────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb in val_loader:
                    Xb = Xb.to(DEVICE)
                    recon = self.model(Xb)
                    val_losses.append(self.criterion(recon, Xb).item())

            t_loss = float(np.mean(tr_losses))
            v_loss = float(np.mean(val_losses))
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            logger.info(f"[AE|{queue_name}] Epoch {epoch:3d} | train={t_loss:.5f} val={v_loss:.5f}")

            if v_loss < best_val:
                best_val = v_loss
                history["best_epoch"] = epoch
                patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        # Save checkpoint
        ckpt = self.model_dir / f"ae_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved autoencoder → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Detector
# ─────────────────────────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Wraps a trained LSTMAutoencoder for real-time anomaly scoring.

    Threshold: 95th percentile of reconstruction errors on normal validation data.
    Anomaly score: mean squared reconstruction error per window.
    """

    def __init__(
        self,
        model:             LSTMAutoencoder,
        threshold_pct:     float = 95.0,
    ):
        self.model         = model.to(DEVICE)
        self.threshold_pct = threshold_pct
        self.threshold:    Optional[float] = None
        self._cal_errors:  Optional[np.ndarray] = None

    def calibrate_threshold(self, X_normal_val: np.ndarray) -> float:
        """Calibrate anomaly threshold on held-out normal data."""
        errors = self._compute_errors(X_normal_val)
        self._cal_errors = errors
        self.threshold   = float(np.percentile(errors, self.threshold_pct))
        logger.info(
            f"AnomalyDetector threshold calibrated | "
            f"pct={self.threshold_pct}% | threshold={self.threshold:.5f} | "
            f"n_cal={len(errors)}"
        )
        return self.threshold

    def _compute_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-window reconstruction MSE."""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for i in range(0, len(X), 64):    # mini-batch for memory
                batch = torch.tensor(X[i:i+64], dtype=torch.float32).to(DEVICE)
                recon = self.model(batch)
                mse   = torch.mean((batch - recon) ** 2, dim=(1, 2))  # (batch,)
                errors.extend(mse.cpu().numpy().tolist())
        return np.array(errors)

    def score(self, x: np.ndarray) -> dict:
        """
        Score a single window for anomaly.
        x: (seq_len, features) or (1, seq_len, features)
        Returns: anomaly_score, is_anomaly, threshold
        """
        if self.threshold is None:
            raise RuntimeError("Call calibrate_threshold() before scoring")

        if x.ndim == 2:
            x = x[np.newaxis, ...]

        errors = self._compute_errors(x)
        score  = float(errors[0])

        return {
            "anomaly_score": round(score, 6),
            "is_anomaly":    score > self.threshold,
            "threshold":     self.threshold,
            "severity":      self._severity(score),
        }

    def score_batch(self, X: np.ndarray) -> dict:
        """Score a batch of windows. Returns per-window scores + summary."""
        if self.threshold is None:
            raise RuntimeError("Call calibrate_threshold() first")

        errors    = self._compute_errors(X)
        is_anomaly = errors > self.threshold
        return {
            "anomaly_scores": errors,
            "is_anomaly":     is_anomaly,
            "anomaly_rate":   float(is_anomaly.mean()),
            "max_score":      float(errors.max()),
            "threshold":      self.threshold,
        }

    def _severity(self, score: float) -> str:
        if self.threshold is None:
            return "unknown"
        ratio = score / self.threshold
        if ratio < 1.0:
            return "normal"
        if ratio < 1.5:
            return "low"
        if ratio < 2.5:
            return "medium"
        return "high"

    @classmethod
    def from_checkpoint(
        cls, path: str, input_size: int = 11, seq_len: int = 12
    ) -> "AnomalyDetector":
        model = LSTMAutoencoder(input_size=input_size, seq_len=seq_len)
        ckpt  = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded autoencoder from {path}")
        return cls(model=model)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration
# ─────────────────────────────────────────────────────────────────────────────
def log_autoencoder_to_mlflow(
    queue_name: str,
    history: dict,
    threshold: float,
    anomaly_rate: float,
    tracking_uri: Optional[str] = None,
):
    """Log autoencoder training metrics to MLflow."""
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("lstm_autoencoder")

        with mlflow.start_run(run_name=f"ae_{queue_name.replace(' ', '_')}"):
            mlflow.log_param("queue_name", queue_name)
            mlflow.log_param("best_epoch", history.get("best_epoch", 0))
            mlflow.log_metric("final_val_loss", history["val_loss"][-1] if history["val_loss"] else 0)
            mlflow.log_metric("anomaly_threshold", threshold)
            mlflow.log_metric("anomaly_rate", anomaly_rate)

        logger.info(f"Autoencoder metrics logged to MLflow for {queue_name}")
    except Exception as e:
        logger.warning(f"MLflow logging skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 2 — LSTM Autoencoder Anomaly Detector Smoke Test")
    print("=" * 60)

    rng      = np.random.default_rng(42)
    seq_len  = 12
    features = 11
    n_normal = 400
    n_test   = 100

    # Normal data (low-variance sine-ish pattern)
    X_normal = rng.normal(0, 1, (n_normal, seq_len, features)).astype(np.float32)
    # Anomalous windows (high-variance spikes)
    X_anomaly = rng.normal(0, 5, (20, seq_len, features)).astype(np.float32)
    X_test    = np.concatenate([
        rng.normal(0, 1, (n_test, seq_len, features)).astype(np.float32),
        X_anomaly,
    ])

    model   = LSTMAutoencoder(input_size=features, seq_len=seq_len)
    trainer = AutoencoderTrainer(model, epochs=5, batch_size=32)
    history = trainer.fit(X_normal, queue_name="test_queue")

    detector = AnomalyDetector(model)
    detector.calibrate_threshold(X_normal[-80:])

    results  = detector.score_batch(X_test)
    # Last 20 are anomalies → should mostly be flagged
    flagged_anomalies = results["is_anomaly"][-20:].sum()
    print(f"\n  Normal windows flagged as anomaly: {results['is_anomaly'][:n_test].sum()}/{n_test}")
    print(f"  Injected anomalies detected:       {flagged_anomalies}/20")
    print(f"  Threshold:                         {results['threshold']:.5f}")
    single = detector.score(X_anomaly[0])
    print(f"  Single anomaly score:              {single['anomaly_score']:.5f} | {single['severity']}")
    print("\n✓ Module 2 (LSTM Autoencoder) — PASSED\n")
