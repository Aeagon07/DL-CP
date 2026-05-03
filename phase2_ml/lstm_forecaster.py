"""
Phase 2 — PyTorch LSTM Volume Forecaster
==========================================
Architecture:
  - Multi-layer bidirectional LSTM
  - Dropout regularization
  - Multi-horizon output heads (1h, 2h, 4h)
  - Uncertainty quantification via Monte Carlo Dropout

Training:
  - Walk-forward cross-validation (no data leakage)
  - MSE + quantile loss for calibrated confidence bands
  - OneCycleLR scheduler with warmup

Inference:
  - Returns point forecast + 80% CI via MC dropout sampling
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"LSTM using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class OperationalSequenceDataset(Dataset):
    """
    Dataset of (sequence, target) pairs for LSTM training.
    X: (seq_len, num_features) — sliding window of feature snapshots
    y: (3,) — volume at 1h, 2h, 4h ahead (normalised)
    """

    def __init__(
        self,
        X: np.ndarray,    # (N, seq_len, features)
        y: np.ndarray,    # (N, 3)   — [vol_1h, vol_2h, vol_4h]
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class LSTMForecaster(nn.Module):
    """
    Bidirectional multi-layer LSTM with multi-horizon output heads.

    Architecture:
      Input (seq_len, batch, features)
      → BiLSTM × num_layers (hidden*2 output)
      → LayerNorm
      → Dropout
      → 3 independent FC heads (1h / 2h / 4h)
    """

    def __init__(
        self,
        input_size:  int = 11,
        hidden_size: int = 128,
        num_layers:  int = 2,
        dropout:     float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        self.num_dirs      = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
            batch_first = True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_dirs)
        self.dropout    = nn.Dropout(dropout)

        fc_in = hidden_size * self.num_dirs

        # Separate forecast heads for each horizon
        self.head_1h = nn.Sequential(
            nn.Linear(fc_in, 64), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
        self.head_2h = nn.Sequential(
            nn.Linear(fc_in, 64), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )
        self.head_4h = nn.Sequential(
            nn.Linear(fc_in, 64), nn.ReLU(), nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, features)
        returns: (batch, 3)  — forecasts for 1h, 2h, 4h
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*dirs)
        last = lstm_out[:, -1, :]           # Take last time step
        last = self.layer_norm(last)
        last = self.dropout(last)

        out_1h = self.head_1h(last)         # (batch, 1)
        out_2h = self.head_2h(last)
        out_4h = self.head_4h(last)

        return torch.cat([out_1h, out_2h, out_4h], dim=1)  # (batch, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Loss
# ─────────────────────────────────────────────────────────────────────────────
class PinballLoss(nn.Module):
    """Pinball / quantile loss for probabilistic forecasting."""

    def __init__(self, quantile: float = 0.5):
        super().__init__()
        self.q = quantile

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = target - pred
        return torch.mean(torch.where(err >= 0, self.q * err, (self.q - 1) * err))


class CombinedLoss(nn.Module):
    """MSE + Pinball loss for point + quantile estimates."""

    def __init__(self):
        super().__init__()
        self.mse     = nn.MSELoss()
        self.q_low   = PinballLoss(0.1)
        self.q_high  = PinballLoss(0.9)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(pred, target) + 0.3 * self.q_low(pred, target) + 0.3 * self.q_high(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class LSTMTrainer:
    def __init__(
        self,
        model: LSTMForecaster,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        model_dir: str = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = CombinedLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        queue_name: str = "default",
    ) -> dict:
        """Train with early stopping. Returns training history dict."""
        train_ds = OperationalSequenceDataset(X_train, y_train)
        val_ds   = OperationalSequenceDataset(X_val,   y_val)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False, num_workers=0)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr,
            steps_per_epoch=len(train_loader), epochs=self.epochs,
        )

        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────
            self.model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                pred = self.model(Xb)
                loss = self.criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            # ── Validate ──────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    pred = self.model(Xb)
                    val_losses.append(self.criterion(pred, yb).item())

            t_loss = float(np.mean(train_losses))
            v_loss = float(np.mean(val_losses))
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            logger.info(f"[{queue_name}] Epoch {epoch:3d} | train={t_loss:.4f} val={v_loss:.4f}")

            # ── Early Stopping ────────────────────────────────
            if v_loss < best_val:
                best_val = v_loss
                history["best_epoch"] = epoch
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        # Save
        ckpt_path = self.model_dir / f"lstm_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt_path)
        logger.info(f"Saved LSTM checkpoint → {ckpt_path}")

        return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference with MC Dropout (Uncertainty Estimation)
# ─────────────────────────────────────────────────────────────────────────────
class LSTMInference:
    """
    Wraps a trained LSTMForecaster for real-time inference.
    Applies MC Dropout (T=30 forward passes) for uncertainty bands.
    """

    def __init__(self, model: LSTMForecaster, mc_samples: int = 30):
        self.model      = model.to(DEVICE)
        self.mc_samples = mc_samples

    @classmethod
    def from_checkpoint(
        cls, path: str, input_size: int = 11, mc_samples: int = 30
    ) -> "LSTMInference":
        model = LSTMForecaster(input_size=input_size)
        ckpt  = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Loaded LSTM from {path}")
        return cls(model=model, mc_samples=mc_samples)

    def predict(self, x: np.ndarray) -> dict:
        """
        x: (seq_len, features) or (1, seq_len, features)
        Returns: {
          'mean': [1h, 2h, 4h],
          'lower_80': [...],
          'upper_80': [...],
        }
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]   # add batch dim

        tensor = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        # Enable dropout for MC sampling
        self._enable_mc_dropout()
        samples = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                pred = self.model(tensor).cpu().numpy()  # (1, 3)
                samples.append(pred[0])

        samples = np.array(samples)  # (mc_samples, 3)
        mean    = samples.mean(axis=0).tolist()
        lower   = np.percentile(samples, 10, axis=0).tolist()
        upper   = np.percentile(samples, 90, axis=0).tolist()

        return {
            "mean":      {"1h": mean[0],  "2h": mean[1],  "4h": mean[2]},
            "lower_80":  {"1h": lower[0], "2h": lower[1], "4h": lower[2]},
            "upper_80":  {"1h": upper[0], "2h": upper[1], "4h": upper[2]},
        }

    def _enable_mc_dropout(self):
        """Enable dropout layers during inference for MC sampling."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
