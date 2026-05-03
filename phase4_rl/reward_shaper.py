"""
Phase 4 — Reward Shaper
========================
Multi-objective reward function for the Appian RL agent.

Reward formula:
    R = -w1 * breach_rate_mean          (primary: minimize SLA breaches)
      + w2 * throughput_norm            (secondary: maximize completions)
      - w3 * utilization_penalty        (tertiary: avoid burnout >95%)
      - w4 * action_magnitude           (regularization: penalize extreme moves)
      + w5 * compliance_bonus           (bonus: reward >90% SLA compliance)
      + improvement_bonus               (+1.0 if breach rate improved vs last step)
      + crash_penalty                   (-5.0 if any queue breaches >80%)

Default weights: w1=5.0, w2=1.0, w3=0.5, w4=0.2, w5=2.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from phase3_simulation.simpy_engine import SimulationResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Reward weight configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RewardWeights:
    breach_penalty:      float = 5.0   # w1 — primary objective
    throughput_bonus:    float = 1.0   # w2
    utilization_penalty: float = 0.5   # w3
    action_magnitude:    float = 0.2   # w4 — regularisation
    compliance_bonus:    float = 2.0   # w5
    improvement_bonus:   float = 1.0   # step improvement reward
    crash_penalty:       float = 5.0   # single-queue >80% breach

    # Thresholds
    burnout_threshold:   float = 0.95  # utilization above this → penalty
    compliance_target:   float = 0.90  # SLA compliance above this → bonus
    crash_threshold:     float = 0.80  # per-queue breach rate → crash penalty


# ─────────────────────────────────────────────────────────────────────────────
# Normalization constants (derived from Phase 3 baseline runs)
# ─────────────────────────────────────────────────────────────────────────────
# Max expected throughput per queue per 8-hour shift (from DEFAULT_QUEUE_CONFIGS)
_MAX_THROUGHPUT_PER_QUEUE = {
    "Document Review":     96.0,   # 12 cases/hr × 8h
    "Compliance Check":    64.0,   # 8  cases/hr × 8h
    "Payment Processing":  160.0,  # 20 cases/hr × 8h
    "Customer Onboarding": 48.0,   # 6  cases/hr × 8h
    "Risk Assessment":     40.0,   # 5  cases/hr × 8h
    "Audit Preparation":   24.0,   # 3  cases/hr × 8h
}
_GLOBAL_MAX_THROUGHPUT = sum(_MAX_THROUGHPUT_PER_QUEUE.values())  # 432


class RewardShaper:
    """
    Computes the scalar reward for one RL step from SimulationResult objects.

    Usage:
        shaper = RewardShaper()
        reward, info = shaper.compute(results, action, prev_breach_rate=0.05)
    """

    def __init__(self, weights: Optional[RewardWeights] = None):
        self.w = weights or RewardWeights()
        self._episode_rewards: List[float] = []

    # ------------------------------------------------------------------
    def compute(
        self,
        results:           List[SimulationResult],
        action:            np.ndarray,
        prev_breach_rate:  float = 0.0,
    ) -> Tuple[float, dict]:
        """
        Compute reward from a list of SimulationResults (one per queue per run).

        Args:
            results:          Flat list of SimulationResult from QueueSimulation.
            action:           The action taken (MultiDiscrete array).
            prev_breach_rate: Breach rate from the previous step (for improvement bonus).

        Returns:
            (reward, info_dict) — scalar reward and diagnostic breakdown.
        """
        if not results:
            return -10.0, {"error": "no_results"}

        # ── Aggregate metrics across all queues / runs ─────────────────
        breach_rates   = [r.breach_rate       for r in results]
        compliances    = [r.sla_compliance_rate for r in results]
        utilizations   = [r.utilization        for r in results]
        throughputs    = [r.throughput_per_hour for r in results]

        mean_breach     = float(np.mean(breach_rates))
        mean_compliance = float(np.mean(compliances))
        mean_util       = float(np.mean(utilizations))
        total_throughput= float(np.sum(throughputs))  # sum across queues

        # ── Per-queue breach rates (for crash detection) ───────────────
        by_queue: Dict[str, List[float]] = {}
        for r in results:
            by_queue.setdefault(r.queue_name, []).append(r.breach_rate)
        queue_p50_breach = {q: float(np.median(v)) for q, v in by_queue.items()}
        max_queue_breach = max(queue_p50_breach.values()) if queue_p50_breach else 0.0

        # ── Reward components ──────────────────────────────────────────
        # 1. Breach penalty (primary)
        breach_component = -self.w.breach_penalty * mean_breach

        # 2. Throughput bonus (normalised to [0, 1])
        throughput_norm = min(1.0, total_throughput / max(_GLOBAL_MAX_THROUGHPUT, 1.0))
        throughput_component = self.w.throughput_bonus * throughput_norm

        # 3. Utilization penalty (only if above burnout threshold)
        util_excess = max(0.0, mean_util - self.w.burnout_threshold)
        util_component = -self.w.utilization_penalty * util_excess

        # 4. Action magnitude regularization
        action_deltas = [abs(int(a) - 2) for a in action]   # 2 = neutral index
        action_mag = float(np.mean(action_deltas)) / 2.0    # normalise to [0,1]
        action_component = -self.w.action_magnitude * action_mag

        # 5. SLA compliance bonus
        compliance_component = 0.0
        if mean_compliance >= self.w.compliance_target:
            compliance_component = self.w.compliance_bonus * (mean_compliance - self.w.compliance_target)

        # 6. Improvement bonus
        improvement_component = 0.0
        if mean_breach < prev_breach_rate - 0.005:   # at least 0.5% improvement
            improvement_component = self.w.improvement_bonus

        # 7. Crash penalty (single queue severely breaching)
        crash_component = 0.0
        if max_queue_breach > self.w.crash_threshold:
            crash_component = -self.w.crash_penalty

        # ── Total reward ───────────────────────────────────────────────
        reward = (
            breach_component
            + throughput_component
            + util_component
            + action_component
            + compliance_component
            + improvement_component
            + crash_component
        )

        self._episode_rewards.append(reward)

        info = {
            "reward_total":         round(reward, 4),
            "breach_component":     round(breach_component, 4),
            "throughput_component": round(throughput_component, 4),
            "util_component":       round(util_component, 4),
            "action_component":     round(action_component, 4),
            "compliance_component": round(compliance_component, 4),
            "improvement_bonus":    round(improvement_component, 4),
            "crash_penalty":        round(crash_component, 4),
            "mean_breach_rate":     round(mean_breach, 4),
            "mean_compliance":      round(mean_compliance, 4),
            "mean_utilization":     round(mean_util, 4),
            "throughput_norm":      round(throughput_norm, 4),
            "max_queue_breach":     round(max_queue_breach, 4),
            "queue_p50_breach":     {q: round(v, 4) for q, v in queue_p50_breach.items()},
        }

        return float(reward), info

    # ------------------------------------------------------------------
    def reset_episode(self):
        """Call at the start of each episode to clear reward history."""
        self._episode_rewards = []

    # ------------------------------------------------------------------
    def episode_summary(self) -> dict:
        """
        Returns statistics over all rewards collected in the current episode.
        Call at episode end (after done=True).
        """
        rewards = self._episode_rewards
        if not rewards:
            return {}
        return {
            "episode_total_reward": round(float(np.sum(rewards)),   3),
            "episode_mean_reward":  round(float(np.mean(rewards)),  3),
            "episode_std_reward":   round(float(np.std(rewards)),   3),
            "episode_min_reward":   round(float(np.min(rewards)),   3),
            "episode_max_reward":   round(float(np.max(rewards)),   3),
            "episode_length":       len(rewards),
        }

    # ------------------------------------------------------------------
    @staticmethod
    def reward_description() -> str:
        """Returns a human-readable explanation of the reward function."""
        return (
            "R = -5.0 × breach_rate"
            " + 1.0 × throughput_norm"
            " - 0.5 × util_excess"
            " - 0.2 × action_magnitude"
            " + 2.0 × compliance_bonus"
            " + 1.0 × improvement_bonus"
            " - 5.0 × crash_penalty"
        )
