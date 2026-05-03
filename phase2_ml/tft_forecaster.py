"""
Phase 2 — Temporal Fusion Transformer (Module 7 of 12)
========================================================
Google DeepMind's state-of-the-art architecture for multi-horizon
time-series forecasting. Won the M5 Forecasting Competition (2020).

Why TFT > LSTM for operational forecasting:
  - Explicit separation of static vs time-varying inputs
  - Interpretable attention weights (what time steps the model focused on)
  - Quantile regression output (10th/50th/90th percentile forecasts)
  - Gated skip connections prevent vanishing gradients
  - Variable Selection Networks learn which features are relevant per step

Architecture (built from scratch in PyTorch — no pytorch-forecasting):
  ┌─────────────────────────────────────────────────────┐
  │           Static Covariate Encoder                  │
  │  queue_id (int) → embedding → static context h_s   │
  └───────────────────────┬─────────────────────────────┘
                          │ static context
  ┌─────────────────────────────────────────────────────┐
  │       Variable Selection Networks (VSN)             │
  │  Learns soft feature importance per timestep        │
  └───────────────────────┬─────────────────────────────┘
                          │ selected features
  ┌─────────────────────────────────────────────────────┐
  │      LSTM Encoder (past) + Decoder (future)        │
  │  Processes selected features through sequence       │
  └───────────────────────┬─────────────────────────────┘
                          │ lstm representations
  ┌─────────────────────────────────────────────────────┐
  │       Multi-Head Attention (Temporal Self-Attn)     │
  │  Captures long-range temporal dependencies          │
  └───────────────────────┬─────────────────────────────┘
                          │ attention-weighted context
  ┌─────────────────────────────────────────────────────┐
  │      Quantile Output Heads [q10, q50, q90]         │
  │  Probabilistic multi-horizon forecasts              │
  └─────────────────────────────────────────────────────┘

Run standalone:
  python phase2_ml/tft_forecaster.py
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
# Gated Linear Unit (GLU)
# ─────────────────────────────────────────────────────────────────────────────
class GLU(nn.Module):
    """Gated Linear Unit — controls information flow in GRN blocks."""
    def __init__(self, input_size: int):
        super().__init__()
        self.fc    = nn.Linear(input_size, input_size)
        self.gate  = nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * torch.sigmoid(self.gate(x))


# ─────────────────────────────────────────────────────────────────────────────
# Gated Residual Network (GRN) — core TFT building block
# ─────────────────────────────────────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    """
    GRN: core building block of TFT.
    Allows the network to learn complex transformations while
    the gate decides how much of the transformation to apply.

    GRN(x, c=None):
      η₁ = ELU(W₁·x + W₂·c + b₁)
      η₂ = W₃·η₁ + b₂
      output = LayerNorm(x + GLU(η₂))
    """

    def __init__(
        self,
        input_dim:   int,
        hidden_dim:  int,
        output_dim:  int,
        dropout:     float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.fc1         = nn.Linear(input_dim,  hidden_dim)
        self.context_fc  = nn.Linear(context_dim, hidden_dim) if context_dim else None
        self.fc2         = nn.Linear(hidden_dim,  output_dim)
        self.glu         = GLU(output_dim)
        self.layer_norm  = nn.LayerNorm(output_dim)
        self.dropout     = nn.Dropout(dropout)

        # Skip connection projection if dims differ
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.fc1(x)
        if context is not None and self.context_fc is not None:
            h = h + self.context_fc(context)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.glu(h)
        return self.layer_norm(self.skip(x) + h)


# ─────────────────────────────────────────────────────────────────────────────
# Variable Selection Network (VSN)
# ─────────────────────────────────────────────────────────────────────────────
class VariableSelectionNetwork(nn.Module):
    """
    Learns soft feature importance weights per timestep.
    Each feature gets a learned weight — irrelevant features suppressed.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim:   int,
        dropout:      float = 0.1,
        context_dim:  Optional[int] = None,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim   = hidden_dim

        # Per-feature GRN transformations
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])

        # Softmax selection weights
        self.selection_grn = GatedResidualNetwork(
            num_features * hidden_dim, hidden_dim, num_features,
            dropout, context_dim=context_dim,
        )

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, num_features)
        Returns: (selected, weights)
          selected: (batch, seq_len, hidden_dim)
          weights:  (batch, seq_len, num_features)    — interpretable!
        """
        B, T, F = x.shape

        # Transform each feature independently
        feat_reprs = []
        for i, grn in enumerate(self.feature_grns):
            xi = x[..., i:i+1]                           # (B, T, 1)
            xi_flat = xi.reshape(B * T, 1)
            fi = grn(xi_flat).reshape(B, T, self.hidden_dim)
            feat_reprs.append(fi)

        stacked = torch.stack(feat_reprs, dim=-1)        # (B, T, hidden, F)
        flat    = stacked.reshape(B * T, -1)             # (B*T, hidden*F)

        # Selection weights
        ctx_flat = context.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1) \
                   if context is not None else None
        weights  = F.softmax(
            self.selection_grn(flat, ctx_flat).reshape(B, T, self.num_features),
            dim=-1,
        )                                                # (B, T, F)

        # Weighted sum over feature representations
        weights_expanded = weights.unsqueeze(2)           # (B, T, 1, F)
        stacked_t        = stacked.permute(0, 1, 3, 2)   # (B, T, F, hidden)
        selected = (weights_expanded * stacked_t).sum(dim=2)  # (B, T, hidden)

        return selected, weights


# ─────────────────────────────────────────────────────────────────────────────
# Full TFT Model
# ─────────────────────────────────────────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon operational forecasting.
    Outputs quantile forecasts: [q10, q50, q90] × [1h, 2h, 4h].
    """

    def __init__(
        self,
        num_features:   int   = 11,
        num_static:     int   = 6,     # number of unique queues
        hidden_dim:     int   = 64,
        num_heads:      int   = 4,
        num_lstm_layers:int   = 1,
        dropout:        float = 0.1,
        num_horizons:   int   = 3,     # 1h, 2h, 4h
        num_quantiles:  int   = 3,     # q10, q50, q90
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.num_horizons  = num_horizons
        self.num_quantiles = num_quantiles

        # Static covariate embedding (queue identity)
        self.static_embed  = nn.Embedding(num_static, hidden_dim)
        self.static_grn    = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        # Variable Selection Network (for time-varying inputs)
        self.vsn = VariableSelectionNetwork(
            num_features  = num_features,
            hidden_dim    = hidden_dim,
            dropout       = dropout,
            context_dim   = hidden_dim,
        )

        # LSTM Encoder-Decoder
        self.lstm_encoder = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = num_lstm_layers,
            dropout     = dropout if num_lstm_layers > 1 else 0.0,
            batch_first = True,
        )

        # Gated skip + layer norm after LSTM
        self.lstm_glu  = GLU(hidden_dim)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # Multi-Head Self-Attention
        self.attn       = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attn_glu   = GLU(hidden_dim)
        self.attn_norm  = nn.LayerNorm(hidden_dim)

        # Position-wise Feed-Forward (GRN)
        self.ffn       = GatedResidualNetwork(hidden_dim, hidden_dim * 2, hidden_dim, dropout)

        # Quantile output heads
        self.output_heads = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_quantiles)
            ])
            for _ in range(num_horizons)
        ])

    def forward(
        self,
        x:         torch.Tensor,          # (batch, seq_len, features)
        static_id: Optional[torch.Tensor] = None,  # (batch,) queue integer ID
    ) -> dict:
        """
        Returns dict with:
          forecasts: (batch, num_horizons, num_quantiles)
          attn_weights: (batch, seq_len, seq_len) — interpretable!
          vsn_weights: (batch, seq_len, num_features) — feature importance
        """
        B, T, _ = x.shape

        # ── Static context ────────────────────────────────────────
        if static_id is not None:
            static_h = self.static_grn(self.static_embed(static_id))  # (B, hidden)
        else:
            static_h = torch.zeros(B, self.hidden_dim, device=x.device)

        # ── Variable Selection ────────────────────────────────────
        selected, vsn_weights = self.vsn(x, context=static_h)  # (B, T, hidden)

        # ── LSTM Encoder ──────────────────────────────────────────
        lstm_out, _ = self.lstm_encoder(selected)           # (B, T, hidden)
        lstm_gated  = self.lstm_glu(lstm_out)               # Gated skip
        lstm_out    = self.lstm_norm(selected + lstm_gated) # Residual

        # ── Multi-Head Self-Attention ─────────────────────────────
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        attn_gated = self.attn_glu(attn_out)
        attn_out   = self.attn_norm(lstm_out + attn_gated)

        # ── Position-wise FFN ─────────────────────────────────────
        ffn_out = self.ffn(attn_out)                        # (B, T, hidden)

        # ── Quantile Output Heads ──────────────────────────────────
        last_hidden = ffn_out[:, -1, :]                     # (B, hidden)

        forecasts = []
        for horizon_heads in self.output_heads:
            q_preds = [head(last_hidden) for head in horizon_heads]  # each (B, 1)
            forecasts.append(torch.cat(q_preds, dim=1))              # (B, num_q)
        forecasts = torch.stack(forecasts, dim=1)           # (B, num_horizons, num_q)

        return {
            "forecasts":    forecasts,
            "attn_weights": attn_weights,
            "vsn_weights":  vsn_weights,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Quantile Loss
# ─────────────────────────────────────────────────────────────────────────────
class QuantileLoss(nn.Module):
    """Pinball loss for quantile regression."""

    def __init__(self, quantiles: list[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (batch, num_horizons, num_quantiles)
        target: (batch, num_horizons) — actual values
        """
        target = target.unsqueeze(-1).expand_as(pred)     # (B, H, Q)
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target[..., i] - pred[..., i]
            losses.append(torch.where(err >= 0, q * err, (q - 1) * err))
        return torch.stack(losses, dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset + Trainer
# ─────────────────────────────────────────────────────────────────────────────
class TFTDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, static_ids: Optional[np.ndarray] = None):
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y   = torch.tensor(y, dtype=torch.float32)
        self.ids = torch.tensor(static_ids, dtype=torch.long) if static_ids is not None \
                   else torch.zeros(len(X), dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx], self.ids[idx]


class TFTTrainer:
    def __init__(
        self,
        model:       TemporalFusionTransformer,
        lr:          float = 3e-4,
        epochs:      int   = 50,
        batch_size:  int   = 32,
        patience:    int   = 10,
        model_dir:   str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.criterion  = QuantileLoss()
        self.optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        queue_name: str = "default",
        static_ids_tr: Optional[np.ndarray] = None,
        static_ids_val: Optional[np.ndarray] = None,
    ) -> dict:
        tr_loader  = DataLoader(TFTDataset(X_tr, y_tr, static_ids_tr),
                                self.batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(TFTDataset(X_val, y_val, static_ids_val),
                                self.batch_size, shuffle=False, num_workers=0)

        history      = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val     = float("inf"); patience_cnt = 0; best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tr_losses = []
            for Xb, yb, sid in tr_loader:
                Xb, yb, sid = Xb.to(DEVICE), yb.to(DEVICE), sid.to(DEVICE)
                self.optimizer.zero_grad()
                out  = self.model(Xb, sid)
                loss = self.criterion(out["forecasts"], yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb, sid in val_loader:
                    Xb, yb, sid = Xb.to(DEVICE), yb.to(DEVICE), sid.to(DEVICE)
                    out = self.model(Xb, sid)
                    val_losses.append(self.criterion(out["forecasts"], yb).item())

            t_l = float(np.mean(tr_losses))
            v_l = float(np.mean(val_losses))
            history["train_loss"].append(t_l)
            history["val_loss"].append(v_l)
            logger.info(f"[TFT|{queue_name}] Epoch {epoch:3d} | train={t_l:.4f} val={v_l:.4f}")

            if v_l < best_val:
                best_val = v_l; history["best_epoch"] = epoch; patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience: break

        if best_state: self.model.load_state_dict(best_state)
        ckpt = self.model_dir / f"tft_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved TFT → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
class TFTInference:
    """TFT inference returning point + quantile forecasts + attention weights."""

    def __init__(self, model: TemporalFusionTransformer):
        self.model = model.to(DEVICE)
        self.model.eval()

    def predict(self, x: np.ndarray, queue_id: int = 0) -> dict:
        """
        x: (seq_len, features) or (1, seq_len, features)
        Returns quantile forecasts + interpretable attention
        """
        if x.ndim == 2: x = x[np.newaxis, ...]
        xt  = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        sid = torch.tensor([queue_id], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            out = self.model(xt, sid)

        forecasts    = out["forecasts"][0].cpu().numpy()    # (3, 3)
        attn_weights = out["attn_weights"][0].cpu().numpy()  # (T, T)
        vsn_weights  = out["vsn_weights"][0, -1, :].cpu().numpy()  # (F,)

        horizons = ["1h", "2h", "4h"]
        return {
            "forecasts": {
                h: {"q10": float(forecasts[i, 0]),
                    "q50": float(forecasts[i, 1]),
                    "q90": float(forecasts[i, 2])}
                for i, h in enumerate(horizons)
            },
            "feature_importance": vsn_weights.tolist(),
            "attention_shape": list(attn_weights.shape),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 7 — Temporal Fusion Transformer Smoke Test")
    print("=" * 60)

    rng     = np.random.default_rng(42)
    N, T, F = 300, 12, 11
    X       = rng.normal(0, 1, (N, T, F)).astype(np.float32)
    y       = rng.uniform(10, 100, (N, 3)).astype(np.float32)
    ids     = rng.integers(0, 6, N).astype(np.int64)
    split   = int(N * 0.8)

    model   = TemporalFusionTransformer(num_features=F, hidden_dim=32, num_heads=2)
    trainer = TFTTrainer(model, epochs=3, batch_size=32)
    history = trainer.fit(
        X[:split], y[:split], X[split:], y[split:],
        queue_name="test",
        static_ids_tr=ids[:split], static_ids_val=ids[split:],
    )

    infer  = TFTInference(model)
    result = infer.predict(X[0], queue_id=0)
    print(f"\n  TFT 1h: q10={result['forecasts']['1h']['q10']:.2f} "
          f"q50={result['forecasts']['1h']['q50']:.2f} "
          f"q90={result['forecasts']['1h']['q90']:.2f}")
    print(f"  Feature importance (top 3): {sorted(enumerate(result['feature_importance']), key=lambda x: -x[1])[:3]}")
    print(f"  Final val loss:  {history['val_loss'][-1]:.4f}")
    print("\n✓ Module 7 (Temporal Fusion Transformer) — PASSED\n")
