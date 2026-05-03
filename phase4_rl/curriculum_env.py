"""
Phase 4 — Curriculum Environment Wrapper
=========================================
Wraps AppianQueueEnv with 3-phase progressive difficulty:

  Phase 1 (Easy):   Baseline arrival rates ±10%, 20 steps/episode
  Phase 2 (Medium): Arrival rates ±25%, faster SLA deadlines (×0.75)
  Phase 3 (Hard):   Arrival spikes up to ×1.5, random mid-episode agent failures

Advances to next phase when mean reward over last 50 episodes
exceeds the configured threshold.

Mirrors the curriculum_learning.py pattern from Phase 2.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

from phase4_rl.queue_env import AppianQueueEnv, EnvConfig
from phase4_rl.reward_shaper import RewardWeights

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Phase definitions
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CurriculumPhase:
    phase_id:         int
    name:             str
    arrival_noise:    float         # ±% randomisation on arrival rates
    sla_multiplier:   float         # multiplier on SLA hours (< 1 = tighter)
    agent_failure_prob: float       # probability of losing 1 agent mid-episode
    max_steps:        int
    advance_threshold: float        # mean reward threshold to advance


CURRICULUM_PHASES: List[CurriculumPhase] = [
    CurriculumPhase(
        phase_id           = 1,
        name               = "easy",
        arrival_noise      = 0.10,
        sla_multiplier     = 1.00,
        agent_failure_prob = 0.00,
        max_steps          = 20,
        advance_threshold  = -3.0,
    ),
    CurriculumPhase(
        phase_id           = 2,
        name               = "medium",
        arrival_noise      = 0.25,
        sla_multiplier     = 0.85,
        agent_failure_prob = 0.05,
        max_steps          = 20,
        advance_threshold  = -1.5,
    ),
    CurriculumPhase(
        phase_id           = 3,
        name               = "hard",
        arrival_noise      = 0.50,
        sla_multiplier     = 0.75,
        agent_failure_prob = 0.10,
        max_steps          = 25,
        advance_threshold  = float("inf"),  # final phase — never auto-advances
    ),
]


class CurriculumEnv(gym.Wrapper):
    """
    Progressive difficulty curriculum wrapper for AppianQueueEnv.

    Usage:
        base_env = AppianQueueEnv()
        env      = CurriculumEnv(base_env)
        obs, _   = env.reset()
        obs, reward, term, trunc, info = env.step(action)

    The wrapper monitors cumulative episode rewards and advances the
    difficulty phase automatically when the agent is performing well.
    """

    def __init__(
        self,
        base_env: Optional[AppianQueueEnv] = None,
        window_size: int = 50,
    ):
        if base_env is None:
            base_env = AppianQueueEnv()
        super().__init__(base_env)

        self._phases         = CURRICULUM_PHASES
        self._phase_idx      = 0            # start at phase 0 (easy)
        self._window_size    = window_size
        self._reward_window: deque = deque(maxlen=window_size)
        self._episode_reward: float = 0.0
        self._episode_count:  int   = 0
        self._phase_history: List[dict] = []

        self._apply_phase()

    # ------------------------------------------------------------------
    @property
    def current_phase(self) -> CurriculumPhase:
        return self._phases[self._phase_idx]

    @property
    def phase_name(self) -> str:
        return self.current_phase.name

    @property
    def n_phase_advances(self) -> int:
        return self._phase_idx

    # ------------------------------------------------------------------
    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # Record episode reward before reset
        if self._episode_count > 0:
            self._reward_window.append(self._episode_reward)
            self._maybe_advance_phase()

        self._episode_reward = 0.0
        self._episode_count += 1

        obs, info = self.env.reset(**kwargs)
        info["curriculum_phase"] = self.current_phase.name
        info["curriculum_phase_id"] = self.current_phase.phase_id
        return obs, info

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward

        # Random mid-episode agent failure (hard phase)
        if self.current_phase.agent_failure_prob > 0:
            if np.random.random() < self.current_phase.agent_failure_prob:
                self._apply_random_agent_failure()
                info["agent_failure"] = True

        info["curriculum_phase"]    = self.current_phase.name
        info["curriculum_phase_id"] = self.current_phase.phase_id
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def get_curriculum_stats(self) -> dict:
        """Returns curriculum progress statistics."""
        window_rewards = list(self._reward_window)
        return {
            "current_phase":     self.current_phase.name,
            "phase_id":          self.current_phase.phase_id,
            "episodes_completed": self._episode_count,
            "phase_advances":    self._phase_idx,
            "window_mean_reward": round(float(np.mean(window_rewards)), 3) if window_rewards else 0.0,
            "advance_threshold": self.current_phase.advance_threshold,
            "phase_history":     self._phase_history,
        }

    # ------------------------------------------------------------------
    def _apply_phase(self):
        """Update base env config for the current curriculum phase."""
        phase = self.current_phase
        cfg = self.env.config
        cfg.arrival_noise = phase.arrival_noise
        cfg.max_steps     = phase.max_steps
        logger.info(
            f"Curriculum → Phase {phase.phase_id} ({phase.name}): "
            f"noise={phase.arrival_noise:.0%}, sla_mult={phase.sla_multiplier:.2f}, "
            f"failure_prob={phase.agent_failure_prob:.0%}"
        )

    def _maybe_advance_phase(self):
        """Advance to next phase if window mean reward exceeds threshold."""
        if self._phase_idx >= len(self._phases) - 1:
            return   # already at hardest phase
        if len(self._reward_window) < self._window_size:
            return   # not enough episodes yet

        window_mean = float(np.mean(self._reward_window))
        threshold   = self.current_phase.advance_threshold

        if window_mean >= threshold:
            self._phase_history.append({
                "from_phase": self.current_phase.name,
                "episode":    self._episode_count,
                "mean_reward": round(window_mean, 3),
            })
            self._phase_idx += 1
            self._reward_window.clear()
            self._apply_phase()
            logger.info(
                f"🎓 Curriculum advance → Phase {self.current_phase.phase_id} "
                f"({self.current_phase.name}) at episode {self._episode_count}"
            )

    def _apply_random_agent_failure(self):
        """Remove 1 agent from a random non-minimal queue (mid-episode)."""
        from phase4_rl.action_space import QUEUE_MIN_AGENTS
        candidates = [
            q for q, n in self.env._current_agents.items()
            if n > QUEUE_MIN_AGENTS.get(q, 1)
        ]
        if candidates:
            q = np.random.choice(candidates)
            self.env._current_agents[q] -= 1
            logger.debug(f"Agent failure: {q} lost 1 agent → {self.env._current_agents[q]}")
