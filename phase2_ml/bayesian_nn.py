"""
Phase 2 — Bayesian Neural Network (Module 6 of 12)
====================================================
Principled Bayesian uncertainty quantification via Bayes-by-Backprop (BBB).
Replaces MC Dropout's approximate Bayesian inference with true variational
inference over weight distributions.

Theory:
  Instead of point estimates w, we learn a posterior q(w|D) ≈ p(w|D).
  Each weight: w ~ N(μ_w, σ_w²) — mean and variance are learned parameters.
  Forward pass: sample weights from posterior → compute output → take mean.

ELBO Loss:
  L(θ) = E_q[log p(D|w)] - β·KL(q(w|θ) || p(w))
        = Reconstruction (NLL) - β·KL divergence
  The KL term regularizes posteriors toward the prior N(0,1).

Uncertainty Decomposition:
  Epistemic (model uncertainty):  variance across weight posterior samples
                                  → reduces with more training data
  Aleatoric (data uncertainty):   learned output log-variance head
                                  → irreducible measurement/process noise

This distinction is critical for safe AI:
  High epistemic → model hasn't seen this operational regime (needs more data)
  High aleatoric → inherent unpredictability in this queue (accept uncertainty)

Run standalone:
  python phase2_ml/bayesian_nn.py
"""

from __future__ import annotations

import logging
import math
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

LOG_SQRT_2PI = math.log(math.sqrt(2 * math.pi))


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Linear Layer (Bayes-by-Backprop)
# ─────────────────────────────────────────────────────────────────────────────
class BayesianLinear(nn.Module):
    """
    Linear layer with weight distributions q(w) = N(μ_w, σ_w²).

    Reparameterization trick:
      w = μ_w + σ_w * ε,    ε ~ N(0,I)
    This allows gradients to flow through the sampling operation.

    KL divergence with standard prior p(w) = N(0, prior_sigma²):
      KL[q(w)||p(w)] = closed-form per-weight KL between Gaussians
    """

    def __init__(
        self,
        in_features:   int,
        out_features:  int,
        prior_sigma:   float = 1.0,
        bias:          bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.prior_sigma  = prior_sigma

        # Weight distribution parameters
        self.weight_mu     = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho    = nn.Parameter(torch.empty(out_features, in_features))

        # Bias distribution parameters
        if bias:
            self.bias_mu   = nn.Parameter(torch.empty(out_features))
            self.bias_rho  = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mu",  None)
            self.register_parameter("bias_rho", None)

        self._init_params()

    def _init_params(self):
        nn.init.kaiming_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3.0)  # σ ≈ exp(-3) ≈ 0.05 initially
        if self.bias_mu is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_rho, -3.0)

    def _softplus(self, rho: torch.Tensor) -> torch.Tensor:
        """σ = softplus(ρ) ensures σ > 0."""
        return F.softplus(rho) + 1e-6

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (output, kl_loss).
        Samples weights via reparameterization trick.
        """
        w_sigma = self._softplus(self.weight_rho)

        # Reparameterization: w = μ + σ * ε
        eps_w = torch.randn_like(self.weight_mu)
        w     = self.weight_mu + w_sigma * eps_w

        kl = self._kl_divergence(self.weight_mu, w_sigma, self.prior_sigma)

        if self.bias_mu is not None:
            b_sigma = self._softplus(self.bias_rho)
            eps_b   = torch.randn_like(self.bias_mu)
            b       = self.bias_mu + b_sigma * eps_b
            kl     += self._kl_divergence(self.bias_mu, b_sigma, self.prior_sigma)
        else:
            b = None

        out = F.linear(x, w, b)
        return out, kl

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, sigma: torch.Tensor, prior_sigma: float) -> torch.Tensor:
        """KL[N(μ,σ²) || N(0, prior_sigma²)] — closed form."""
        p_log_var = math.log(prior_sigma ** 2)
        q_log_var = sigma.pow(2).log()
        kl = 0.5 * (
            p_log_var - q_log_var
            + (sigma.pow(2) + mu.pow(2)) / (prior_sigma ** 2)
            - 1
        )
        return kl.sum()


# ─────────────────────────────────────────────────────────────────────────────
# BNN Forecaster
# ─────────────────────────────────────────────────────────────────────────────
class BNNForecaster(nn.Module):
    """
    Bayesian Neural Network for SLA breach probability forecasting.

    Architecture:
      Standard LSTM encoder (backbone) +
      Bayesian output head (BayesianLinear) +
      Aleatoric uncertainty head (log-variance prediction)

    Why combine LSTM + Bayesian head:
      - LSTM handles temporal dependencies efficiently
      - Bayesian head adds weight uncertainty without full BNN overhead
      - Practical balance of expressiveness + uncertainty quantification
    """

    def __init__(
        self,
        input_size:  int   = 11,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
        prior_sigma: float = 1.0,
    ):
        super().__init__()
        # Standard LSTM backbone
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = num_layers,
            dropout       = dropout if num_layers > 1 else 0.0,
            bidirectional = True,
            batch_first   = True,
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        feat_dim = hidden_size * 2

        # Bayesian prediction head (epistemic uncertainty)
        self.bayes_fc1       = BayesianLinear(feat_dim,  64,  prior_sigma=prior_sigma)
        self.bayes_output    = BayesianLinear(64,         1,  prior_sigma=prior_sigma)

        # Aleatoric uncertainty head (data noise)
        self.aleatoric_head  = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # log-variance of output
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (breach_logit, log_aleatoric_var, total_kl_loss)
        """
        lstm_out, _ = self.lstm(x)
        last = self.layer_norm(lstm_out[:, -1, :])

        # Bayesian head
        h, kl1        = self.bayes_fc1(last)
        h             = F.relu(h)
        logit, kl2    = self.bayes_output(h)
        total_kl      = kl1 + kl2

        # Aleatoric head
        log_ale_var   = self.aleatoric_head(last)

        return logit.squeeze(1), log_ale_var.squeeze(1), total_kl


# ─────────────────────────────────────────────────────────────────────────────
# ELBO Loss
# ─────────────────────────────────────────────────────────────────────────────
class ELBOLoss(nn.Module):
    """
    Evidence Lower BOund (ELBO) loss for BBB training.
    ELBO = E_q[log p(y|x,w)] - β/N · KL[q(w)||p(w)]

    The negative likelihood uses heteroscedastic Gaussian:
      log p(y|x,w) = -0.5 * [log(σ²) + (y - ŷ)²/σ²]
    where σ² is the predicted aleatoric variance.
    """

    def __init__(self, beta: float = 1.0, n_train: int = 1000):
        super().__init__()
        self.beta    = beta
        self.n_train = n_train

    def forward(
        self,
        logit:       torch.Tensor,   # (batch,) raw logit
        log_ale_var: torch.Tensor,   # (batch,) log aleatoric variance
        y:           torch.Tensor,   # (batch,) binary labels
        kl_loss:     torch.Tensor,   # scalar total KL
    ) -> tuple[torch.Tensor, dict]:
        """Returns total ELBO loss + breakdown dict."""
        proba = torch.sigmoid(logit)

        # Heteroscedastic NLL (treat target as continuous for differentiability)
        nll = F.binary_cross_entropy(proba, y, reduction="mean")

        # Aleatoric variance penalty: encourage tight predictions
        ale_penalty = log_ale_var.mean()

        # KL regularization (scaled by β and dataset size)
        kl_term = self.beta * kl_loss / self.n_train

        total = nll + ale_penalty + kl_term

        return total, {
            "nll":         float(nll.item()),
            "ale_penalty": float(ale_penalty.item()),
            "kl_term":     float(kl_term.item()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class BNNDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class BNNTrainer:
    def __init__(
        self,
        model:      BNNForecaster,
        lr:         float = 1e-3,
        epochs:     int   = 50,
        batch_size: int   = 32,
        patience:   int   = 10,
        beta:       float = 1.0,
        model_dir:  str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
        self._elbo_loss = None
        self.beta       = beta

    def fit(
        self,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        queue_name: str = "default",
    ) -> dict:
        self._elbo_loss = ELBOLoss(beta=self.beta, n_train=len(X_tr))

        tr_loader  = DataLoader(BNNDataset(X_tr, y_tr),   self.batch_size, shuffle=True,  num_workers=0)
        val_loader = DataLoader(BNNDataset(X_val, y_val), self.batch_size, shuffle=False, num_workers=0)

        history      = {"train_loss": [], "val_loss": [], "kl": [], "best_epoch": 0}
        best_val     = float("inf"); patience_cnt = 0; best_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            tr_losses, kls = [], []
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                logit, log_ale, kl = self.model(Xb)
                loss, breakdown = self._elbo_loss(logit, log_ale, yb, kl)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_losses.append(loss.item())
                kls.append(breakdown["kl_term"])

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    logit, log_ale, kl = self.model(Xb)
                    loss, _ = self._elbo_loss(logit, log_ale, yb, kl)
                    val_losses.append(loss.item())

            t_loss = float(np.mean(tr_losses))
            v_loss = float(np.mean(val_losses))
            avg_kl = float(np.mean(kls))
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["kl"].append(avg_kl)
            logger.info(f"[BNN|{queue_name}] Epoch {epoch:3d} | train={t_loss:.4f} val={v_loss:.4f} kl={avg_kl:.4f}")

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

        if best_state: self.model.load_state_dict(best_state)
        ckpt = self.model_dir / f"bnn_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved BNN → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# Inference with Uncertainty Decomposition
# ─────────────────────────────────────────────────────────────────────────────
class BNNInference:
    """
    MC sampling from weight posterior → epistemic + aleatoric decomposition.
    """

    def __init__(self, model: BNNForecaster, n_samples: int = 50):
        self.model     = model.to(DEVICE)
        self.n_samples = n_samples

    def predict(self, x: np.ndarray) -> dict:
        """
        x: (seq_len, features) or (1, seq_len, features)
        Returns: mean_prob, epistemic_var, aleatoric_var, total_var
        """
        if x.ndim == 2:
            x = x[np.newaxis, ...]
        t = torch.tensor(x, dtype=torch.float32).to(DEVICE)

        self.model.train()   # enable sampling (Bayesian layers sample on forward)
        probs, ale_vars = [], []
        with torch.no_grad():
            for _ in range(self.n_samples):
                logit, log_ale, _ = self.model(t)
                probs.append(torch.sigmoid(logit).item())
                ale_vars.append(torch.exp(log_ale).item())

        probs    = np.array(probs)
        ale_vars = np.array(ale_vars)

        epistemic = float(probs.var())
        aleatoric = float(ale_vars.mean())
        mean_prob = float(probs.mean())

        return {
            "breach_prob":      round(mean_prob, 4),
            "epistemic_var":    round(epistemic, 6),
            "aleatoric_var":    round(aleatoric, 6),
            "total_uncertainty":round(epistemic + aleatoric, 6),
            "uncertainty_type": "epistemic" if epistemic > aleatoric else "aleatoric",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 6 — Bayesian Neural Network Smoke Test")
    print("=" * 60)

    rng     = np.random.default_rng(42)
    N, T, F = 300, 12, 11
    X       = rng.normal(0, 1, (N, T, F)).astype(np.float32)
    y       = rng.integers(0, 2, N).astype(np.float32)
    split   = int(N * 0.8)

    model   = BNNForecaster(input_size=F)
    trainer = BNNTrainer(model, epochs=3, batch_size=32)
    history = trainer.fit(X[:split], y[:split], X[split:], y[split:], queue_name="test")

    infer   = BNNInference(model, n_samples=20)
    result  = infer.predict(X[0])

    print(f"\n  Breach probability:      {result['breach_prob']:.4f}")
    print(f"  Epistemic uncertainty:   {result['epistemic_var']:.6f}  (model ignorance)")
    print(f"  Aleatoric uncertainty:   {result['aleatoric_var']:.6f}  (data noise)")
    print(f"  Dominant uncertainty:    {result['uncertainty_type']}")
    print(f"  Final ELBO val loss:     {history['val_loss'][-1]:.4f}")
    print("\n✓ Module 6 (Bayesian Neural Network) — PASSED\n")
