"""
Phase 4 — RL Inference Engine
===============================
Loads a trained PPO model and generates human-readable staffing
recommendations from current queue state observations.

Runs fully standalone (no Kafka/Redis required by default).
Optional Redis write via --use-feature-store flag in runner.py.

Key classes:
    RLRecommendation    — dataclass with actions, confidence, narrative
    RLInferenceEngine   — loads model, builds obs, returns recommendation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from phase3_simulation.simpy_engine import DEFAULT_QUEUE_CONFIGS
from phase4_rl.action_space import (
    ActionSpace,
    QUEUE_NAMES,
    QUEUE_DEFAULT_AGENTS,
)
from phase4_rl.queue_env import (
    AppianQueueEnv,
    EnvConfig,
    OBS_DIM,
    N_FEATURES_PER_QUEUE,
    _OBS_MAX,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RLRecommendation:
    """
    Output of one inference call.

    Attributes:
        timestamp:                   When the recommendation was generated
        actions:                     {queue_name: agent_delta}
        current_agents:              {queue_name: current_agent_count}
        recommended_agents:          {queue_name: recommended_agent_count}
        action_probabilities:        Softmax probability of chosen action per queue
        predicted_breach_reduction:  Estimated breach rate improvement (0–1)
        confidence:                  Mean action probability (proxy for certainty)
        narrative:                   Human-readable recommendation text
        queue_states:                Raw input state snapshot
    """
    timestamp:                   datetime
    actions:                     Dict[str, int]
    current_agents:              Dict[str, int]
    recommended_agents:          Dict[str, int]
    action_probabilities:        List[float]
    predicted_breach_reduction:  float
    confidence:                  float
    narrative:                   str
    queue_states:                Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp":                   self.timestamp.isoformat(),
            "actions":                     self.actions,
            "current_agents":              self.current_agents,
            "recommended_agents":          self.recommended_agents,
            "confidence":                  round(self.confidence, 3),
            "predicted_breach_reduction":  round(self.predicted_breach_reduction, 3),
            "narrative":                   self.narrative,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference engine
# ─────────────────────────────────────────────────────────────────────────────
class RLInferenceEngine:
    """
    Loads a trained PPO model and generates staffing recommendations.

    Usage (standalone):
        engine = RLInferenceEngine("phase4_rl/checkpoints/best_model")
        rec    = engine.recommend()          # uses synthetic baseline state
        print(rec.narrative)

    Usage (with custom queue states):
        states = {
            "Payment Processing": {"wip": 45, "utilization": 0.92,
                                   "breach_rate": 0.25, "avg_wait_hours": 1.8,
                                   "throughput_per_hour": 15.0},
            ...
        }
        rec = engine.recommend(queue_states=states)

    Usage (with Redis FeatureStore):
        engine = RLInferenceEngine("path/to/model", use_feature_store=True)
        rec    = engine.recommend()
    """

    def __init__(
        self,
        model_path: str,
        use_feature_store: bool = False,
        current_agents: Optional[Dict[str, int]] = None,
    ):
        from stable_baselines3 import PPO
        self._model = PPO.load(model_path)
        self._use_feature_store = use_feature_store
        self._current_agents = current_agents or dict(QUEUE_DEFAULT_AGENTS)
        self._store = None

        if use_feature_store:
            try:
                from phase1_pipeline.feature_store import FeatureStore
                self._store = FeatureStore()
                logger.info("Connected to FeatureStore (Redis)")
            except Exception as e:
                logger.warning(f"FeatureStore unavailable: {e}. Using synthetic state.")
                self._use_feature_store = False

        logger.info(f"RLInferenceEngine loaded model from {model_path}")

    # ------------------------------------------------------------------
    def recommend(
        self,
        queue_states: Optional[Dict[str, dict]] = None,
    ) -> RLRecommendation:
        """
        Generate a staffing recommendation.

        Args:
            queue_states: Optional dict of {queue_name: metric_dict}.
                          If None, uses FeatureStore or baseline synthetic state.

        Returns:
            RLRecommendation with actions, confidence, and narrative.
        """
        # 1. Get queue states
        if queue_states is None:
            queue_states = self._get_queue_states()

        # 2. Build observation vector
        obs = self._build_observation(queue_states)

        # 3. Get model action + probabilities
        action, action_probs = self._predict_with_probs(obs)

        # 4. Decode action
        deltas = ActionSpace.decode(action)
        recommended_agents = ActionSpace.apply(deltas, self._current_agents)

        # 5. Estimate breach reduction (heuristic from action)
        pred_breach_reduction = self._estimate_breach_reduction(
            queue_states, deltas
        )

        # 6. Compute confidence
        confidence = float(np.mean([max(p) for p in action_probs])) if action_probs else 0.5

        # 7. Generate narrative
        narrative = self._build_narrative(deltas, recommended_agents, queue_states)

        rec = RLRecommendation(
            timestamp                  = datetime.now(),
            actions                    = deltas,
            current_agents             = dict(self._current_agents),
            recommended_agents         = recommended_agents,
            action_probabilities       = [float(max(p)) for p in action_probs] if action_probs else [],
            predicted_breach_reduction = pred_breach_reduction,
            confidence                 = confidence,
            narrative                  = narrative,
            queue_states               = queue_states,
        )

        # 8. Write to FeatureStore if enabled
        if self._store:
            try:
                self._store.set_rl_recommendation(rec.to_dict())
                logger.info("Recommendation written to FeatureStore")
            except Exception as e:
                logger.warning(f"FeatureStore write failed: {e}")

        return rec

    # ------------------------------------------------------------------
    def run_inference_loop(self, interval_seconds: int = 300):
        """
        Continuously generate recommendations every interval_seconds.
        Press Ctrl+C to stop.
        """
        import time
        logger.info(f"Starting inference loop (interval={interval_seconds}s). Ctrl+C to stop.")
        while True:
            try:
                rec = self.recommend()
                print(f"\n[{rec.timestamp.strftime('%H:%M:%S')}] {rec.narrative}")
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                logger.info("Inference loop stopped.")
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _get_queue_states(self) -> Dict[str, dict]:
        """Get queue states from FeatureStore or generate synthetic baseline."""
        if self._use_feature_store and self._store:
            try:
                snapshots = self._store.get_all_queues_latest()
                if snapshots:
                    return {
                        q: {
                            "wip":               s.wip,
                            "utilization":       s.utilization,
                            "breach_rate":       getattr(s, "breach_rate", 0.05),
                            "avg_wait_hours":    getattr(s, "avg_wait_hours", 0.5),
                            "throughput_per_hour": getattr(s, "throughput", 10.0),
                        }
                        for q, s in snapshots.items()
                    }
            except Exception as e:
                logger.warning(f"FeatureStore read failed: {e}. Using synthetic state.")

        # Synthetic baseline state (from DEFAULT_QUEUE_CONFIGS)
        states = {}
        for name, cfg in DEFAULT_QUEUE_CONFIGS.items():
            states[name] = {
                "wip":                float(cfg["base_arrival"]) * 0.5,
                "utilization":        0.70,
                "breach_rate":        0.05,
                "avg_wait_hours":     float(cfg["sla_hours"]) * 0.2,
                "throughput_per_hour": float(cfg["base_arrival"]) * 0.85,
            }
        return states

    def _build_observation(self, queue_states: Dict[str, dict]) -> np.ndarray:
        """Convert queue state dict into 48-dim normalized observation."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        for qi, q in enumerate(QUEUE_NAMES):
            base = qi * N_FEATURES_PER_QUEUE
            state = queue_states.get(q, {})
            cfg   = DEFAULT_QUEUE_CONFIGS.get(q, {})

            obs[base + 0] = state.get("wip", 10.0)                   / _OBS_MAX["wip"]
            obs[base + 1] = state.get("utilization", 0.7)            / _OBS_MAX["utilization"]
            obs[base + 2] = state.get("breach_rate", 0.05)           / _OBS_MAX["breach_rate"]
            obs[base + 3] = state.get("avg_wait_hours", 0.5)         / _OBS_MAX["avg_wait"]
            obs[base + 4] = float(cfg.get("base_arrival", 10))       / _OBS_MAX["arrival_rate"]
            obs[base + 5] = self._current_agents.get(q, 5)           / _OBS_MAX["agents"]
            obs[base + 6] = float(cfg.get("sla_hours", 8))           / _OBS_MAX["sla_hours"]
            obs[base + 7] = state.get("throughput_per_hour", 10.0)   / _OBS_MAX["throughput"]

        return np.clip(obs, 0.0, 1.0)

    def _predict_with_probs(self, obs: np.ndarray):
        """Get action and per-queue action probabilities from the policy."""
        action, _ = self._model.predict(obs, deterministic=True)
        # Try to get action probabilities from the policy
        try:
            import torch
            obs_tensor = self._model.policy.obs_to_tensor(obs)[0]
            with torch.no_grad():
                dist = self._model.policy.get_distribution(obs_tensor)
                # For MultiDiscrete, dist is a list of Categorical distributions
                probs = [d.probs.cpu().numpy().tolist() for d in dist.distribution]
        except Exception:
            probs = []
        return action, probs

    def _estimate_breach_reduction(
        self,
        queue_states: Dict[str, dict],
        deltas: Dict[str, int],
    ) -> float:
        """
        Heuristic estimate of breach reduction from adding agents to stressed queues.
        High-utilization queues benefit most from additional agents.
        """
        reduction = 0.0
        for q, delta in deltas.items():
            if delta > 0:
                util = queue_states.get(q, {}).get("utilization", 0.7)
                br   = queue_states.get(q, {}).get("breach_rate", 0.05)
                # Adding agents to a stressed queue reduces breach proportionally
                reduction += delta * util * br * 0.3
            elif delta < 0:
                # Removing agents may increase breach
                util = queue_states.get(q, {}).get("utilization", 0.7)
                if util > 0.8:
                    reduction -= abs(delta) * 0.05
        return round(min(1.0, max(-1.0, reduction)), 3)

    def _build_narrative(
        self,
        deltas: Dict[str, int],
        recommended_agents: Dict[str, int],
        queue_states: Dict[str, dict],
    ) -> str:
        """Generate a human-readable recommendation narrative."""
        increases = [(q, d) for q, d in deltas.items() if d > 0]
        decreases = [(q, d) for q, d in deltas.items() if d < 0]
        unchanged = [(q, d) for q, d in deltas.items() if d == 0]

        if not increases and not decreases:
            return ("✅ Current staffing is optimal. "
                    "No agent reallocation recommended at this time.")

        parts = []

        if increases:
            for q, d in sorted(increases, key=lambda x: -x[1]):
                br   = queue_states.get(q, {}).get("breach_rate", 0) * 100
                curr = self._current_agents.get(q, 0)
                new  = recommended_agents.get(q, curr)
                parts.append(
                    f"➕ {q}: Add {d} agent{'s' if abs(d)>1 else ''} "
                    f"({curr}→{new}) — breach rate {br:.1f}%"
                )

        if decreases:
            for q, d in sorted(decreases, key=lambda x: x[1]):
                curr = self._current_agents.get(q, 0)
                new  = recommended_agents.get(q, curr)
                parts.append(
                    f"➖ {q}: Remove {abs(d)} agent{'s' if abs(d)>1 else ''} "
                    f"({curr}→{new}) — capacity available"
                )

        if unchanged:
            hold_queues = ", ".join(q for q, _ in unchanged)
            parts.append(f"⏸  Hold: {hold_queues}")

        return " | ".join(parts)
