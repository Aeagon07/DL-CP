"""
Phase 2 — TabTransformer for Tabular Breach Classification (Module 8 of 12)
============================================================================
Applies Transformer self-attention over FEATURE EMBEDDINGS (not time steps).
Each feature in the tabular operational snapshot is tokenized into a D-dim
embedding, then Transformer blocks learn feature-feature interactions.

Why TabTransformer vs XGBoost:
  - XGBoost treats features independently (no higher-order interactions)
  - TabTransformer attends over feature embeddings → learns which features
    interact (e.g. wip_count × utilization_rate → breach signal)
  - Differentiable → can be fine-tuned end-to-end with neural components

Architecture:
  Tabular row (num_features,)
    → Feature Tokenizer: embed each feature into (num_features, d_model)
    → [CLS] token prepended → (num_features+1, d_model)
    → N × Transformer Block (multi-head attention + FFN + LayerNorm)
    → CLS token output → MLP classifier → breach_probability

This is the FT-Transformer (Feature Tokenizer + Transformer) variant,
similar to the architecture in "Revisiting Deep Learning Models for Tabular Data"
(Gorishniy et al., 2021).

Run standalone:
  python phase2_ml/tab_transformer.py
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
# Feature Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
class FeatureTokenizer(nn.Module):
    """
    Transforms each scalar feature into a D-dimensional learned token.

    For continuous features: token_i = x_i * W_i + b_i
    where W_i ∈ R^D is a learned embedding vector per feature.

    This gives the Transformer a consistent D-dim input regardless of
    feature scale or type.
    """

    def __init__(self, num_features: int, d_model: int = 128):
        super().__init__()
        self.num_features = num_features
        self.d_model       = d_model

        # Per-feature weight and bias vectors
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias   = nn.Parameter(torch.zeros(num_features, d_model))

        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, num_features)
        Returns: (batch, num_features, d_model)
        """
        # x[:, i] scalar × W[i] vector + b[i] vector
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────────────────────────────────────
class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block:
      LayerNorm → MultiHeadAttention → Residual
      LayerNorm → FFN → Residual
    Pre-norm (vs post-norm) trains more stably on tabular data.
    """

    def __init__(
        self,
        d_model:     int   = 128,
        num_heads:   int   = 8,
        ffn_dim:     int   = 256,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn  = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model)
        Returns: (output, attention_weights)
        """
        # Self-attention with pre-norm
        normed  = self.norm1(x)
        attn_out, attn_w = self.attn(normed, normed, normed)
        x = x + attn_out

        # FFN with pre-norm
        x = x + self.ffn(self.norm2(x))
        return x, attn_w


# ─────────────────────────────────────────────────────────────────────────────
# Full TabTransformer Model
# ─────────────────────────────────────────────────────────────────────────────
class TabTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer for tabular SLA breach classification.

    Takes a single operational snapshot (one row of features) and predicts
    breach probability for a given horizon.
    """

    def __init__(
        self,
        num_features:   int   = 35,    # number of tabular features
        d_model:        int   = 128,
        num_heads:      int   = 8,
        num_blocks:     int   = 4,
        ffn_dim:        int   = 256,
        dropout:        float = 0.1,
        cls_hidden:     int   = 64,
    ):
        super().__init__()
        self.num_features = num_features

        # Feature tokenization
        self.tokenizer  = FeatureTokenizer(num_features, d_model)

        # Learnable [CLS] token
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer blocks
        self.blocks     = nn.ModuleList([
            TransformerBlock(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Classification head on [CLS] token
        self.cls_head   = nn.Sequential(
            nn.Linear(d_model, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (batch, num_features) — flat tabular row
        Returns: breach_prob, attention_maps
        """
        B = x.size(0)

        # Tokenize features
        tokens = self.tokenizer(x)                     # (B, F, d_model)

        # Prepend [CLS] token
        cls    = self.cls_token.expand(B, -1, -1)      # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, F+1, d_model)

        # Pass through Transformer blocks
        attn_maps = []
        for block in self.blocks:
            tokens, attn_w = block(tokens)
            attn_maps.append(attn_w.detach().cpu())

        tokens = self.final_norm(tokens)

        # Classification on [CLS] token (index 0)
        cls_output  = tokens[:, 0, :]                   # (B, d_model)
        breach_prob = self.cls_head(cls_output)          # (B, 1)

        return {
            "breach_prob": breach_prob.squeeze(1),      # (B,)
            "attn_maps":   attn_maps,                    # list of (B, F+1, F+1)
        }

    def get_feature_attention(self, attn_maps: list) -> np.ndarray:
        """
        Extract mean feature attention from last block (interpretable).
        Returns: (num_features,) importance scores
        """
        last_attn   = attn_maps[-1].mean(0)              # (F+1, F+1)
        cls_to_feat = last_attn[0, 1:].numpy()           # CLS → feature weights
        return cls_to_feat / (cls_to_feat.sum() + 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class TabDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class TabTransformerTrainer:
    """
    Train TabTransformer on tabular breach features.
    Same feature set as XGBoost → enables direct comparison.
    """

    def __init__(
        self,
        model:      TabTransformer,
        lr:         float = 3e-4,
        epochs:     int   = 50,
        batch_size: int   = 64,
        patience:   int   = 10,
        model_dir:  str   = "models/",
    ):
        self.model      = model.to(DEVICE)
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
        queue_name: str = "default",
        horizon_hours: int = 1,
    ) -> dict:
        tr_loader  = DataLoader(TabDataset(X_tr,  y_tr),  self.batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(TabDataset(X_val, y_val), self.batch_size, shuffle=False, num_workers=0)

        history  = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val = float("inf"); patience_cnt = 0; best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tr_losses = []
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                out  = self.model(Xb)
                loss = self.criterion(out["breach_prob"], yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    out = self.model(Xb)
                    val_losses.append(self.criterion(out["breach_prob"], yb).item())

            t_l = float(np.mean(tr_losses))
            v_l = float(np.mean(val_losses))
            history["train_loss"].append(t_l)
            history["val_loss"].append(v_l)
            logger.info(f"[TabTF|{queue_name}|{horizon_hours}h] Epoch {epoch:3d} | train={t_l:.4f} val={v_l:.4f}")

            if v_l < best_val:
                best_val = v_l; history["best_epoch"] = epoch; patience_cnt = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience: break

        if best_state: self.model.load_state_dict(best_state)
        ckpt = self.model_dir / f"tabtf_{queue_name.replace(' ', '_')}_h{horizon_hours}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved TabTransformer → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
class TabTransformerInference:
    """TabTransformer inference with feature attention extraction."""

    def __init__(self, model: TabTransformer):
        self.model = model.to(DEVICE)
        self.model.eval()

    def predict(self, x: np.ndarray) -> dict:
        """
        x: (num_features,) or (1, num_features) — single tabular row
        Returns: breach_prob, feature_attention
        """
        if x.ndim == 1: x = x[np.newaxis, :]
        xt = torch.tensor(x, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            out = self.model(xt)

        feat_attn = self.model.get_feature_attention(out["attn_maps"])
        return {
            "breach_prob":       round(float(out["breach_prob"][0].item()), 4),
            "feature_attention": feat_attn.tolist(),
        }

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Returns breach probabilities for a batch of rows."""
        loader = DataLoader(TabDataset(X, np.zeros(len(X))), batch_size=64, num_workers=0)
        probs  = []
        self.model.eval()
        with torch.no_grad():
            for Xb, _ in loader:
                out = self.model(Xb.to(DEVICE))
                probs.extend(out["breach_prob"].cpu().numpy().tolist())
        return np.array(probs)

    @classmethod
    def from_checkpoint(cls, path: str, num_features: int = 35) -> "TabTransformerInference":
        model = TabTransformer(num_features=num_features)
        ckpt  = torch.load(path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        return cls(model)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration
# ─────────────────────────────────────────────────────────────────────────────
def log_tabtransformer_to_mlflow(
    queue_name: str, horizon_hours: int, history: dict,
    feature_names: list[str], top_attn: np.ndarray,
    tracking_uri: Optional[str] = None,
):
    try:
        import mlflow
        if tracking_uri: mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("tab_transformer")
        with mlflow.start_run(run_name=f"tabtf_{queue_name.replace(' ', '_')}_h{horizon_hours}"):
            mlflow.log_param("queue_name",     queue_name)
            mlflow.log_param("horizon_hours",  horizon_hours)
            mlflow.log_param("best_epoch",     history.get("best_epoch", 0))
            if history["val_loss"]:
                mlflow.log_metric("final_val_loss",  history["val_loss"][-1])
        logger.info(f"TabTransformer metrics logged for {queue_name} h{horizon_hours}")
    except Exception as e:
        logger.warning(f"MLflow skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 8 — TabTransformer Smoke Test")
    print("=" * 60)

    rng  = np.random.default_rng(42)
    N, F = 300, 25
    X    = rng.normal(0, 1, (N, F)).astype(np.float32)
    y    = rng.integers(0, 2, N).astype(np.float32)
    split = int(N * 0.8)

    model    = TabTransformer(num_features=F, d_model=32, num_heads=2, num_blocks=2, ffn_dim=64)
    trainer  = TabTransformerTrainer(model, epochs=3, batch_size=32)
    history  = trainer.fit(X[:split], y[:split], X[split:], y[split:], queue_name="test")

    infer   = TabTransformerInference(model)
    result  = infer.predict(X[0])
    top3    = sorted(enumerate(result["feature_attention"]), key=lambda x: -x[1])[:3]

    print(f"\n  Breach probability:   {result['breach_prob']:.4f}")
    print(f"  Top-3 feature attn:   {top3}")
    print(f"  Final val loss:       {history['val_loss'][-1]:.4f}")
    print("\n✓ Module 8 (TabTransformer) — PASSED\n")
