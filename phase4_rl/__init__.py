"""
Phase 4 — Reinforcement Learning Auto-Optimizer
================================================
Top-level package. Exposes the primary public API.

Quick start (training):
    from phase4_rl import AppianQueueEnv, PPOTrainer
    from phase4_rl.ppo_agent import PPOConfig

    env     = AppianQueueEnv()
    trainer = PPOTrainer(PPOConfig(total_timesteps=200_000))
    model   = trainer.train()

Quick start (inference):
    from phase4_rl import RLInferenceEngine
    engine = RLInferenceEngine("phase4_rl/checkpoints/best_model")
    rec    = engine.recommend(queue_states)
    print(rec.narrative)

CLI:
    python -m phase4_rl.runner demo
    python -m phase4_rl.runner train
    python -m phase4_rl.runner eval --model phase4_rl/checkpoints/best_model
    python -m phase4_rl.runner infer --model phase4_rl/checkpoints/best_model
"""

from phase4_rl.action_space import ActionSpace
from phase4_rl.reward_shaper import RewardShaper, RewardWeights
from phase4_rl.queue_env import AppianQueueEnv, EnvConfig
from phase4_rl.curriculum_env import CurriculumEnv
from phase4_rl.ppo_agent import PPOTrainer, PPOConfig
from phase4_rl.rl_inference import RLInferenceEngine, RLRecommendation

__all__ = [
    "ActionSpace",
    "RewardShaper", "RewardWeights",
    "AppianQueueEnv", "EnvConfig",
    "CurriculumEnv",
    "PPOTrainer", "PPOConfig",
    "RLInferenceEngine", "RLRecommendation",
]
