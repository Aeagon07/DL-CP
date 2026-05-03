"""
Phase 2 — Multi-Task Learner (Module 3 of 12)
===============================================
Joint neural network with two output heads trained simultaneously:
  Head A: SLA Breach probability (binary classification, BCE loss)
  Head B: Volume forecast at 1h/2h/4h horizons (regression, MSE loss)

Why Multi-Task Learning works:
  Shared Bi-LSTM encoder learns representations that are useful for BOTH
  tasks. Breach events correlate with volume surges — the regression task
  regularizes the encoder and prevents overfitting on sparse breach labels.
  Fewer parameters than two separate models. Better generalization.

Architecture:
  Input (batch, seq_len, features)
    → Shared BiLSTM [2 layers, hidden=128, bidirectional]
    → LayerNorm + Dropout
    ├── Classification Head: FC(256→64→1) + Sigmoid → breach_prob
    └── Regression Head:     FC(256→64→3)            → [vol_1h, vol_2h, vol_4h]
  Loss = BCE(breach_pred, y_breach) + λ·MSE(vol_pred, y_vol)
  Adaptive λ via GradNorm-style loss balancing

Run standalone:
  python phase2_ml/multitask_learner.py
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
class MultiTaskDataset(Dataset):
    """Dataset yielding (sequence, breach_label, volume_targets)."""

    def __init__(
        self,
        X:        np.ndarray,  # (N, seq_len, features)
        y_breach: np.ndarray,  # (N,) binary
        y_volume: np.ndarray,  # (N, 3) - [vol_1h, vol_2h, vol_4h]
    ):
        self.X        = torch.tensor(X,        dtype=torch.float32)
        self.y_breach = torch.tensor(y_breach, dtype=torch.float32)
        self.y_volume = torch.tensor(y_volume, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_breach[idx], self.y_volume[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskLSTM(nn.Module):
    """
    Shared Bi-LSTM encoder with dual output heads.

    Classification head: breach probability
    Regression head:     volume at 1h, 2h, 4h
    """

    def __init__(
        self,
        input_size:  int   = 11,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        dir_factor = 2  # bidirectional

        # Shared encoder
        self.shared_lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
            batch_first   = True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * dir_factor)
        self.dropout    = nn.Dropout(dropout)

        feat_dim = hidden_size * dir_factor

        # ── Classification Head (breach probability) ──────────────
        self.cls_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # ── Regression Head (volume forecasts) ───────────────────
        self.reg_head = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 3),          # [vol_1h, vol_2h, vol_4h]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, features)
        Returns: (breach_prob, volume_forecasts)
          breach_prob:      (batch, 1)  — P(breach)
          volume_forecasts: (batch, 3)  — [1h, 2h, 4h]
        """
        lstm_out, _ = self.shared_lstm(x)    # (batch, seq_len, hidden*2)
        last        = lstm_out[:, -1, :]     # last timestep
        last        = self.layer_norm(last)
        last        = self.dropout(last)

        breach_prob = self.cls_head(last)    # (batch, 1)
        vol_pred    = self.reg_head(last)    # (batch, 3)

        return breach_prob.squeeze(1), vol_pred


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Loss Balancer (GradNorm-inspired)
# ─────────────────────────────────────────────────────────────────────────────
class AdaptiveLossBalancer:
    """
    Dynamically balances classification and regression losses.
    Tracks relative loss magnitudes and adjusts λ so neither task dominates.
    """

    def __init__(self, alpha: float = 0.5, ema_decay: float = 0.98):
        self.alpha     = alpha
        self.ema_decay = ema_decay
        self._ema_cls  = 1.0
        self._ema_reg  = 1.0
        self._lambda   = 0.5     # initial weighting

    def update(self, loss_cls: float, loss_reg: float) -> float:
        """Update EMA and recompute λ. Returns current λ."""
        self._ema_cls = self.ema_decay * self._ema_cls + (1 - self.ema_decay) * loss_cls
        self._ema_reg = self.ema_decay * self._ema_reg + (1 - self.ema_decay) * loss_reg

        # Normalize: balance relative contribution
        total = self._ema_cls + self._ema_reg + 1e-8
        self._lambda = float(self._ema_reg / total)  # more λ when regression is harder
        return self._lambda

    @property
    def lambda_weight(self) -> float:
        return self._lambda


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskTrainer:
    """
    Joint training loop for breach classification + volume regression.
    Uses adaptive loss balancing to prevent one task from dominating.
    """

    def __init__(
        self,
        model:       MultiTaskLSTM,
        lr:          float = 1e-3,
        epochs:      int   = 50,
        batch_size:  int   = 32,
        patience:    int   = 10,
        lambda_init: float = 0.5,
        model_dir:   str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.cls_criterion = nn.BCELoss()
        self.reg_criterion = nn.MSELoss()
        self.optimizer     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.balancer      = AdaptiveLossBalancer()

    def fit(
        self,
        X_tr:          np.ndarray,
        y_breach_tr:   np.ndarray,
        y_volume_tr:   np.ndarray,
        X_val:         np.ndarray,
        y_breach_val:  np.ndarray,
        y_volume_val:  np.ndarray,
        queue_name:    str = "default",
    ) -> dict:
        tr_ds  = MultiTaskDataset(X_tr,  y_breach_tr,  y_volume_tr)
        val_ds = MultiTaskDataset(X_val, y_breach_val, y_volume_val)

        tr_loader  = DataLoader(tr_ds,  batch_size=self.batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.lr,
            steps_per_epoch=len(tr_loader), epochs=self.epochs,
        )

        history     = {"train_loss": [], "val_loss": [], "lambda_history": [], "best_epoch": 0}
        best_val    = float("inf")
        patience_cnt = 0
        best_state  = None

        for epoch in range(1, self.epochs + 1):
            # ── Train ─────────────────────────────────────────────
            self.model.train()
            epoch_loss, epoch_cls, epoch_reg = [], [], []
            for Xb, y_cls, y_reg in tr_loader:
                Xb   = Xb.to(DEVICE)
                y_cls = y_cls.to(DEVICE)
                y_reg = y_reg.to(DEVICE)

                self.optimizer.zero_grad()
                pred_cls, pred_reg = self.model(Xb)

                loss_cls = self.cls_criterion(pred_cls, y_cls)
                loss_reg = self.reg_criterion(pred_reg, y_reg)

                lam  = self.balancer.update(loss_cls.item(), loss_reg.item())
                loss = loss_cls + lam * loss_reg

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

                epoch_loss.append(loss.item())
                epoch_cls.append(loss_cls.item())
                epoch_reg.append(loss_reg.item())

            # ── Validate ──────────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, y_cls, y_reg in val_loader:
                    Xb    = Xb.to(DEVICE)
                    y_cls = y_cls.to(DEVICE)
                    y_reg = y_reg.to(DEVICE)
                    pred_cls, pred_reg = self.model(Xb)
                    lc = self.cls_criterion(pred_cls, y_cls)
                    lr = self.reg_criterion(pred_reg, y_reg)
                    val_losses.append((lc + self.balancer.lambda_weight * lr).item())

            t_loss = float(np.mean(epoch_loss))
            v_loss = float(np.mean(val_losses))
            lam    = self.balancer.lambda_weight
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["lambda_history"].append(lam)

            logger.info(
                f"[MTL|{queue_name}] Epoch {epoch:3d} | "
                f"train={t_loss:.4f} val={v_loss:.4f} | "
                f"cls={np.mean(epoch_cls):.4f} reg={np.mean(epoch_reg):.4f} λ={lam:.3f}"
            )

            if v_loss < best_val:
                best_val  = v_loss
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

        ckpt = self.model_dir / f"mtl_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved MTL model → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskInference:
    """Wraps trained MultiTaskLSTM for joint breach + volume inference."""

    def __init__(self, model: MultiTaskLSTM):
        self.model = model.to(DEVICE)
        self.model.eval()

    def predict(self, x: np.ndarray) -> dict:
        """
        x: (seq_len, features) or (1, seq_len, features)
        Returns: breach_prob, volume_1h/2h/4h
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        t = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            p_breach, p_vol = self.model(t)

        return {
            "breach_prob": round(float(p_breach[0].item()), 4),
            "volume": {
                "1h": round(float(p_vol[0, 0].item()), 3),
                "2h": round(float(p_vol[0, 1].item()), 3),
                "4h": round(float(p_vol[0, 2].item()), 3),
            },
        }

    @classmethod
    def from_checkpoint(cls, path: str, input_size: int = 11) -> "MultiTaskInference":
        model = MultiTaskLSTM(input_size=input_size)
        ckpt  = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        return cls(model)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration
# ─────────────────────────────────────────────────────────────────────────────
def log_multitask_to_mlflow(
    queue_name: str,
    history: dict,
    tracking_uri: Optional[str] = None,
):
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("multitask_learning")
        with mlflow.start_run(run_name=f"mtl_{queue_name.replace(' ', '_')}"):
            mlflow.log_param("queue_name", queue_name)
            mlflow.log_param("best_epoch", history.get("best_epoch", 0))
            if history["val_loss"]:
                mlflow.log_metric("final_val_loss", history["val_loss"][-1])
            if history.get("lambda_history"):
                mlflow.log_metric("final_lambda", history["lambda_history"][-1])
        logger.info(f"MTL metrics logged for {queue_name}")
    except Exception as e:
        logger.warning(f"MLflow skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 3 — Multi-Task Learner Smoke Test")
    print("=" * 60)

    rng    = np.random.default_rng(42)
    N, T, F = 300, 12, 11

    X        = rng.normal(0, 1, (N, T, F)).astype(np.float32)
    y_breach = rng.integers(0, 2, N).astype(np.float32)
    y_vol    = rng.uniform(10, 100, (N, 3)).astype(np.float32)

    split    = int(N * 0.8)
    model    = MultiTaskLSTM(input_size=F)
    trainer  = MultiTaskTrainer(model, epochs=3, batch_size=32)
    history  = trainer.fit(
        X[:split], y_breach[:split], y_vol[:split],
        X[split:], y_breach[split:], y_vol[split:],
        queue_name="test",
    )
    infer   = MultiTaskInference(model)
    result  = infer.predict(X[0])
    print(f"\n  Breach prob: {result['breach_prob']:.4f}")
    print(f"  Volume 1h:   {result['volume']['1h']:.2f}")
    print(f"  Volume 2h:   {result['volume']['2h']:.2f}")
    print(f"  Volume 4h:   {result['volume']['4h']:.2f}")
    print(f"  Final λ:     {history['lambda_history'][-1]:.4f}")
    print("\n✓ Module 3 (Multi-Task Learner) — PASSED\n")
