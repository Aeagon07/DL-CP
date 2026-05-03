"""
Phase 2 — Curriculum Learning (Module 4 of 12)
================================================
Trains the LSTM by progressively increasing sample difficulty,
addressing class imbalance more elegantly than SMOTE or re-weighting.

Curriculum Strategy:
  1. Score each training sample by difficulty (XGBoost uncertainty proxy)
  2. Phase 1 (ep 1–20):  feed easy + medium samples (model builds baseline)
  3. Phase 2 (ep 21–40): introduce hard borderline samples gradually
  4. Phase 3 (ep 41+):   uniform sampling (all difficulty levels)

Difficulty Scoring:
  - Easy:   XGBoost prediction confidence > 0.85 (clear breach or no-breach)
  - Medium: 0.60–0.85 confidence
  - Hard:   < 0.60 confidence (borderline / ambiguous cases)

Why curriculum > SMOTE:
  - No synthetic data generation (preserves real temporal structure)
  - Model sees realistic hard cases rather than interpolated noise
  - Can combine with any base model (LSTM, TFT, Neural ODE)

Run standalone:
  python phase2_ml/curriculum_learning.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty Levels
# ─────────────────────────────────────────────────────────────────────────────
EASY   = 0
MEDIUM = 1
HARD   = 2


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty Scorer
# ─────────────────────────────────────────────────────────────────────────────
class DifficultyScorer:
    """
    Scores training samples by difficulty using XGBoost prediction confidence.

    Confidence = |p̂(y=1) - 0.5| * 2  (0 = max uncertainty, 1 = max certainty)
    Thresholds:
      confidence > 0.7  → EASY   (model is very sure → clear example)
      0.2 <= conf <= 0.7 → MEDIUM
      confidence < 0.2  → HARD   (model is unsure → borderline case)
    """

    def __init__(
        self,
        easy_threshold: float = 0.7,
        hard_threshold: float = 0.2,
    ):
        self.easy_threshold = easy_threshold
        self.hard_threshold = hard_threshold

    def score_from_proba(self, proba: np.ndarray) -> np.ndarray:
        """
        proba: P(breach=1) for each sample, shape (N,)
        Returns: difficulty array of ints (0=EASY, 1=MEDIUM, 2=HARD)
        """
        confidence = np.abs(proba - 0.5) * 2    # 0=uncertain, 1=certain
        difficulty = np.full(len(proba), MEDIUM, dtype=int)
        difficulty[confidence >= self.easy_threshold] = EASY
        difficulty[confidence <= self.hard_threshold] = HARD
        return difficulty

    def score_from_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Fallback when XGBoost model not available.
        Uses class imbalance: minority class (breach) = HARD by default.
        """
        breach_rate    = float(y.mean())
        difficulty     = np.full(len(y), MEDIUM, dtype=int)
        difficulty[y == 0] = EASY if breach_rate < 0.3 else MEDIUM
        difficulty[y == 1] = HARD if breach_rate < 0.15 else MEDIUM
        return difficulty

    def score_summary(self, difficulty: np.ndarray) -> dict:
        """Return distribution of difficulty levels."""
        return {
            "easy":   int((difficulty == EASY).sum()),
            "medium": int((difficulty == MEDIUM).sum()),
            "hard":   int((difficulty == HARD).sum()),
            "total":  len(difficulty),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Schedule
# ─────────────────────────────────────────────────────────────────────────────
class CurriculumSchedule:
    """
    Defines the mix of easy/medium/hard samples per training epoch.

    Phase 1 (ep 1–phase1_end):    easy_pct % easy, (1-easy_pct) % medium
    Phase 2 (phase1_end–phase2_end): gradual introduction of hard samples
    Phase 3 (phase2_end+):         uniform sampling
    """

    def __init__(
        self,
        phase1_end: int = 20,
        phase2_end: int = 40,
        phase1_easy_pct: float = 0.8,
        phase2_hard_pct_max: float = 0.3,
    ):
        self.phase1_end          = phase1_end
        self.phase2_end          = phase2_end
        self.phase1_easy_pct     = phase1_easy_pct
        self.phase2_hard_pct_max = phase2_hard_pct_max

    def get_mix(self, epoch: int) -> tuple[float, float, float]:
        """
        Returns (easy_pct, medium_pct, hard_pct) for given epoch.
        """
        if epoch <= self.phase1_end:
            e = self.phase1_easy_pct
            m = 1 - e
            h = 0.0
        elif epoch <= self.phase2_end:
            # Linear interpolation into Phase 2
            progress = (epoch - self.phase1_end) / max(1, self.phase2_end - self.phase1_end)
            h = min(self.phase2_hard_pct_max, progress * self.phase2_hard_pct_max)
            e = max(0.4, self.phase1_easy_pct * (1 - progress))
            m = 1 - e - h
        else:
            # Phase 3: uniform
            e, m, h = 1/3, 1/3, 1/3

        # Normalize
        total = e + m + h
        return e / total, m / total, h / total

    def describe(self, epoch: int) -> str:
        e, m, h = self.get_mix(epoch)
        phase = 1 if epoch <= self.phase1_end else (2 if epoch <= self.phase2_end else 3)
        return f"Phase {phase} | Easy={e:.0%} Medium={m:.0%} Hard={h:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Sampler
# ─────────────────────────────────────────────────────────────────────────────
class CurriculumSampler(Sampler):
    """
    Custom PyTorch Sampler that applies curriculum mixing per epoch.
    Stratified sampling from each difficulty stratum.
    """

    def __init__(
        self,
        difficulty:  np.ndarray,        # (N,) difficulty per sample
        batch_size:  int = 32,
        schedule:    Optional[CurriculumSchedule] = None,
    ):
        self.difficulty  = difficulty
        self.batch_size  = batch_size
        self.schedule    = schedule or CurriculumSchedule()
        self._epoch      = 1

        self._easy_idx   = np.where(difficulty == EASY)[0]
        self._medium_idx = np.where(difficulty == MEDIUM)[0]
        self._hard_idx   = np.where(difficulty == HARD)[0]

        logger.info(
            f"CurriculumSampler | "
            f"easy={len(self._easy_idx)} "
            f"medium={len(self._medium_idx)} "
            f"hard={len(self._hard_idx)}"
        )

    def set_epoch(self, epoch: int):
        """Update current epoch — call at start of each training epoch."""
        self._epoch = epoch

    def __iter__(self):
        e_pct, m_pct, h_pct = self.schedule.get_mix(self._epoch)
        n = len(self.difficulty)

        n_easy   = max(1, int(n * e_pct)) if len(self._easy_idx)   > 0 else 0
        n_medium = max(1, int(n * m_pct)) if len(self._medium_idx) > 0 else 0
        n_hard   = max(0, n - n_easy - n_medium) if len(self._hard_idx) > 0 else 0

        rng     = np.random.default_rng()
        indices = []

        if n_easy > 0 and len(self._easy_idx) > 0:
            indices.extend(rng.choice(self._easy_idx,   size=n_easy,   replace=True).tolist())
        if n_medium > 0 and len(self._medium_idx) > 0:
            indices.extend(rng.choice(self._medium_idx, size=n_medium, replace=True).tolist())
        if n_hard > 0 and len(self._hard_idx) > 0:
            indices.extend(rng.choice(self._hard_idx,   size=n_hard,   replace=True).tolist())

        rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.difficulty)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset for Curriculum Training
# ─────────────────────────────────────────────────────────────────────────────
class CurriculumDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum Trainer
# ─────────────────────────────────────────────────────────────────────────────
class CurriculumTrainer:
    """
    Wraps any classification model with curriculum learning.
    Compatible with: LSTMForecaster classification head, MultiTaskLSTM, etc.
    """

    def __init__(
        self,
        model:          nn.Module,
        scorer:         DifficultyScorer,
        schedule:       Optional[CurriculumSchedule] = None,
        lr:             float = 1e-3,
        epochs:         int   = 60,
        batch_size:     int   = 32,
        patience:       int   = 12,
        model_dir:      str   = "models/",
    ):
        self.model      = model.to(DEVICE)
        self.scorer     = scorer
        self.schedule   = schedule or CurriculumSchedule()
        self.lr         = lr
        self.epochs     = epochs
        self.batch_size = batch_size
        self.patience   = patience
        self.model_dir  = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def fit(
        self,
        X_train:   np.ndarray,
        y_train:   np.ndarray,
        X_val:     np.ndarray,
        y_val:     np.ndarray,
        proba_train: Optional[np.ndarray] = None,
        queue_name:  str = "default",
    ) -> dict:
        """
        proba_train: XGBoost probabilities for difficulty scoring (optional).
                     If None, falls back to label-based difficulty.
        """
        # Score difficulty
        if proba_train is not None:
            difficulty = self.scorer.score_from_proba(proba_train)
        else:
            difficulty = self.scorer.score_from_labels(y_train)

        summary = self.scorer.score_summary(difficulty)
        logger.info(
            f"[{queue_name}] Curriculum difficulty: "
            f"easy={summary['easy']} medium={summary['medium']} hard={summary['hard']}"
        )

        train_ds   = CurriculumDataset(X_train, y_train)
        val_ds     = CurriculumDataset(X_val,   y_val)
        curriculum = CurriculumSampler(difficulty, batch_size=self.batch_size, schedule=self.schedule)

        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

        history      = {
            "train_loss": [], "val_loss": [], "curriculum_phase": [], "best_epoch": 0
        }
        best_val     = float("inf")
        patience_cnt = 0
        best_state   = None

        for epoch in range(1, self.epochs + 1):
            curriculum.set_epoch(epoch)
            tr_loader = DataLoader(train_ds, batch_sampler=None, sampler=curriculum,
                                   batch_size=self.batch_size, num_workers=0)

            # ── Train ─────────────────────────────────────────────
            self.model.train()
            tr_losses = []
            for Xb, yb in tr_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                self.optimizer.zero_grad()
                out  = self.model(Xb)

                # Handle both single-output and multi-output models
                if isinstance(out, tuple):
                    pred = out[0]  # breach head from MTL model
                else:
                    pred = out.squeeze(1) if out.dim() > 1 else out

                loss = self.criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                tr_losses.append(loss.item())

            # ── Validate ──────────────────────────────────────────
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                    out = self.model(Xb)
                    if isinstance(out, tuple):
                        pred = out[0]
                    else:
                        pred = out.squeeze(1) if out.dim() > 1 else out
                    val_losses.append(self.criterion(pred, yb).item())

            t_loss = float(np.mean(tr_losses))
            v_loss = float(np.mean(val_losses))
            phase_desc = self.schedule.describe(epoch)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["curriculum_phase"].append(phase_desc)

            logger.info(
                f"[CL|{queue_name}] Epoch {epoch:3d} | "
                f"train={t_loss:.4f} val={v_loss:.4f} | {phase_desc}"
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

        ckpt = self.model_dir / f"curriculum_{queue_name.replace(' ', '_')}.pt"
        torch.save({"model_state": self.model.state_dict(), "history": history}, ckpt)
        logger.info(f"Saved curriculum model → {ckpt}")
        return history


# ─────────────────────────────────────────────────────────────────────────────
# MLflow Integration
# ─────────────────────────────────────────────────────────────────────────────
def log_curriculum_to_mlflow(
    queue_name: str,
    history: dict,
    difficulty_summary: dict,
    tracking_uri: Optional[str] = None,
):
    try:
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("curriculum_learning")
        with mlflow.start_run(run_name=f"cl_{queue_name.replace(' ', '_')}"):
            mlflow.log_param("queue_name",   queue_name)
            mlflow.log_param("n_easy",       difficulty_summary.get("easy", 0))
            mlflow.log_param("n_medium",     difficulty_summary.get("medium", 0))
            mlflow.log_param("n_hard",       difficulty_summary.get("hard", 0))
            mlflow.log_param("best_epoch",   history.get("best_epoch", 0))
            if history["val_loss"]:
                mlflow.log_metric("final_val_loss", history["val_loss"][-1])
        logger.info(f"Curriculum metrics logged for {queue_name}")
    except Exception as e:
        logger.warning(f"MLflow skipped: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Smoke Test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("\n" + "=" * 60)
    print("  MODULE 4 — Curriculum Learning Smoke Test")
    print("=" * 60)

    rng  = np.random.default_rng(42)
    N, T, F = 400, 12, 11
    X   = rng.normal(0, 1, (N, T, F)).astype(np.float32)
    y   = rng.integers(0, 2, N).astype(np.float32)

    # Simulate XGBoost probabilities for difficulty scoring
    proba = np.clip(y * 0.6 + rng.normal(0, 0.2, N), 0, 1).astype(np.float32)

    # Simple classifier for testing
    class QuickLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(F, 32, batch_first=True)
            self.fc   = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(h[-1]).squeeze(1)

    scorer   = DifficultyScorer()
    diff     = scorer.score_from_proba(proba)
    summary  = scorer.score_summary(diff)
    print(f"\n  Difficulty distribution: {summary}")

    schedule = CurriculumSchedule(phase1_end=2, phase2_end=4)
    model    = QuickLSTM()
    trainer  = CurriculumTrainer(model, scorer, schedule, epochs=5, batch_size=32)
    split    = int(N * 0.8)
    history  = trainer.fit(X[:split], y[:split], X[split:], y[split:],
                           proba_train=proba[:split], queue_name="test")

    print(f"\n  Phase log sample: {history['curriculum_phase'][0]}")
    print(f"  Best epoch:       {history['best_epoch']}")
    print("\n✓ Module 4 (Curriculum Learning) — PASSED\n")
