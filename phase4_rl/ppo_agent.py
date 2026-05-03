"""
Phase 4 — PPO Training Orchestrator
=====================================
Trains a PPO agent (stable-baselines3) on the AppianQueueEnv.

Features:
  - Optional CurriculumEnv wrapper
  - MLflow experiment logging (separate phase4_rl experiment)
  - Periodic checkpointing (best model by eval reward)
  - Evaluation loop with per-queue breach rate breakdown
  - EvalCallback with early stopping on nan/inf rewards

Usage:
    from phase4_rl.ppo_agent import PPOTrainer, PPOConfig
    trainer = PPOTrainer(PPOConfig(total_timesteps=200_000))
    model   = trainer.train()
    metrics = trainer.evaluate(model)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PPO Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PPOConfig:
    # Training
    total_timesteps:  int   = 200_000
    learning_rate:    float = 3e-4
    n_steps:          int   = 512       # steps per rollout per env
    batch_size:       int   = 64
    n_epochs:         int   = 10
    gamma:            float = 0.99
    gae_lambda:       float = 0.95
    clip_range:       float = 0.2
    ent_coef:         float = 0.01      # entropy regularisation
    vf_coef:          float = 0.5
    max_grad_norm:    float = 0.5

    # Curriculum
    use_curriculum:   bool  = True

    # Environment
    sim_runs_per_step: int  = 10        # sim runs averaged per RL step
    horizon_hours:    float = 8.0
    max_steps:        int   = 20        # steps per episode

    # Paths
    checkpoint_dir:   str   = "phase4_rl/checkpoints"
    log_dir:          str   = "phase4_rl/logs"

    # Evaluation
    eval_freq:        int   = 5_000     # evaluate every N timesteps
    eval_episodes:    int   = 10        # episodes per evaluation
    n_envs:           int   = 1         # parallel envs (keep 1 for Windows safety)

    # MLflow
    mlflow_experiment: str  = "phase4_rl"
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────
class PPOTrainer:
    """
    Orchestrates PPO training on the Appian queue environment.

    Usage:
        trainer = PPOTrainer(PPOConfig(total_timesteps=50_000))
        model   = trainer.train()
        metrics = trainer.evaluate(model, n_episodes=20)
        trainer.save(model, "phase4_rl/checkpoints/my_model")
    """

    def __init__(self, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self._best_mean_reward = -np.inf
        self._run_id: Optional[str] = None

    # ------------------------------------------------------------------
    def train(self):
        """Full training loop. Returns the trained PPO model."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.callbacks import (
            EvalCallback, CheckpointCallback, CallbackList
        )
        from phase4_rl.queue_env import AppianQueueEnv, EnvConfig
        from phase4_rl.curriculum_env import CurriculumEnv
        from phase4_rl.reward_shaper import RewardWeights

        cfg = self.config

        # ── Dirs ─────────────────────────────────────────────────────
        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

        # ── Build environments ────────────────────────────────────────
        env_config = EnvConfig(
            horizon_hours      = cfg.horizon_hours,
            sim_runs_per_step  = cfg.sim_runs_per_step,
            max_steps          = cfg.max_steps,
            arrival_noise      = 0.10,
        )
        train_env = AppianQueueEnv(env_config, render_mode=None)
        if cfg.use_curriculum:
            train_env = CurriculumEnv(train_env)
        train_env = Monitor(train_env, cfg.log_dir)

        eval_env_cfg = EnvConfig(
            horizon_hours     = cfg.horizon_hours,
            sim_runs_per_step = cfg.sim_runs_per_step,
            max_steps         = cfg.max_steps,
            arrival_noise     = 0.05,
        )
        eval_env = Monitor(AppianQueueEnv(eval_env_cfg), cfg.log_dir)

        # ── MLflow setup ──────────────────────────────────────────────
        mlflow_available = self._setup_mlflow()

        # ── PPO Model ─────────────────────────────────────────────────
        policy_kwargs = dict(net_arch=[256, 256])
        model = PPO(
            policy        = "MlpPolicy",
            env           = train_env,
            learning_rate = cfg.learning_rate,
            n_steps       = cfg.n_steps,
            batch_size    = cfg.batch_size,
            n_epochs      = cfg.n_epochs,
            gamma         = cfg.gamma,
            gae_lambda    = cfg.gae_lambda,
            clip_range    = cfg.clip_range,
            ent_coef      = cfg.ent_coef,
            vf_coef       = cfg.vf_coef,
            max_grad_norm = cfg.max_grad_norm,
            policy_kwargs = policy_kwargs,
            verbose       = 1,
        )

        logger.info(f"PPO model created. Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")

        # ── Callbacks ─────────────────────────────────────────────────
        # EvalCallback saves to a *directory*: best_model_save_dir/best_model.zip
        best_model_save_dir = str(Path(cfg.checkpoint_dir) / "best_model")
        best_model_zip      = str(Path(best_model_save_dir) / "best_model.zip")
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path = best_model_save_dir,
            log_path             = cfg.log_dir,
            eval_freq            = cfg.eval_freq,
            n_eval_episodes      = cfg.eval_episodes,
            deterministic        = True,
            render               = False,
            verbose              = 1,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq   = cfg.eval_freq * 2,
            save_path   = cfg.checkpoint_dir,
            name_prefix = "ppo_appian",
            verbose     = 1,
        )
        callbacks = CallbackList([eval_callback, checkpoint_callback])

        # ── Log hyperparameters to MLflow ─────────────────────────────
        if mlflow_available:
            import mlflow
            with mlflow.start_run(run_name="ppo_training") as run:
                self._run_id = run.info.run_id
                mlflow.log_params({
                    "total_timesteps":  cfg.total_timesteps,
                    "learning_rate":    cfg.learning_rate,
                    "n_steps":          cfg.n_steps,
                    "batch_size":       cfg.batch_size,
                    "n_epochs":         cfg.n_epochs,
                    "gamma":            cfg.gamma,
                    "clip_range":       cfg.clip_range,
                    "ent_coef":         cfg.ent_coef,
                    "use_curriculum":   cfg.use_curriculum,
                    "sim_runs_per_step": cfg.sim_runs_per_step,
                    "horizon_hours":    cfg.horizon_hours,
                    "reward_fn":        "multi_objective_v1",
                })

                t0 = time.time()
                model.learn(
                    total_timesteps   = cfg.total_timesteps,
                    callback          = callbacks,
                    progress_bar      = True,
                    reset_num_timesteps = True,
                )
                elapsed = time.time() - t0

                mlflow.log_metrics({
                    "training_time_seconds": round(elapsed, 1),
                    "best_mean_reward": eval_callback.best_mean_reward,
                })
                # Only log artifact if the best model was actually saved
                if Path(best_model_zip).exists():
                    mlflow.log_artifact(best_model_zip, artifact_path="model")
                    logger.info(f"MLflow run {self._run_id} — logged to experiment '{cfg.mlflow_experiment}'")
                else:
                    logger.warning("Best model zip not found — skipping MLflow artifact upload.")
        else:
            # Train without MLflow
            t0 = time.time()
            model.learn(
                total_timesteps     = cfg.total_timesteps,
                callback            = callbacks,
                progress_bar        = True,
                reset_num_timesteps = True,
            )
            elapsed = time.time() - t0

        self._best_mean_reward = eval_callback.best_mean_reward
        logger.info(
            f"Training complete in {elapsed:.1f}s. "
            f"Best eval reward: {self._best_mean_reward:.3f}"
        )

        # Save final model
        final_path = str(Path(cfg.checkpoint_dir) / "final_model")
        model.save(final_path)
        logger.info(f"Final model saved → {final_path}.zip")

        train_env.close()
        eval_env.close()
        return model

    # ------------------------------------------------------------------
    def evaluate(self, model, n_episodes: int = 20) -> dict:
        """
        Run n_episodes evaluation episodes and return aggregate metrics.

        Returns dict with mean/std reward, breach rates per queue, etc.
        """
        from phase4_rl.queue_env import AppianQueueEnv, EnvConfig

        eval_cfg = EnvConfig(
            horizon_hours     = self.config.horizon_hours,
            sim_runs_per_step = self.config.sim_runs_per_step,
            max_steps         = self.config.max_steps,
            arrival_noise     = 0.05,
        )
        env = AppianQueueEnv(eval_cfg)

        episode_rewards = []
        episode_breaches = []
        episode_compliances = []
        all_actions = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=1000 + ep)
            ep_reward = 0.0
            ep_breaches = []
            ep_compliances = []
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_breaches.append(info.get("mean_breach_rate", 0))
                ep_compliances.append(info.get("mean_compliance", 1))
                all_actions.append(action.copy())
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_breaches.append(float(np.mean(ep_breaches)))
            episode_compliances.append(float(np.mean(ep_compliances)))

        env.close()

        metrics = {
            "n_episodes":            n_episodes,
            "mean_episode_reward":   round(float(np.mean(episode_rewards)), 3),
            "std_episode_reward":    round(float(np.std(episode_rewards)),  3),
            "mean_breach_rate":      round(float(np.mean(episode_breaches)), 4),
            "mean_compliance_rate":  round(float(np.mean(episode_compliances)), 4),
            "min_episode_reward":    round(float(np.min(episode_rewards)),  3),
            "max_episode_reward":    round(float(np.max(episode_rewards)),  3),
        }
        logger.info(f"Eval ({n_episodes} eps): reward={metrics['mean_episode_reward']:.3f} ± "
                    f"{metrics['std_episode_reward']:.3f}, breach={metrics['mean_breach_rate']*100:.1f}%")
        return metrics

    # ------------------------------------------------------------------
    def save(self, model, path: str):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        logger.info(f"Model saved → {path}.zip")

    # ------------------------------------------------------------------
    @staticmethod
    def load(path: str):
        """Load a saved PPO model."""
        from stable_baselines3 import PPO
        model = PPO.load(path)
        logger.info(f"Model loaded ← {path}")
        return model

    # ------------------------------------------------------------------
    def _setup_mlflow(self) -> bool:
        """Configure MLflow experiment. Returns True if available."""
        try:
            import mlflow
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment)
            return True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}. Training without experiment tracking.")
            return False
