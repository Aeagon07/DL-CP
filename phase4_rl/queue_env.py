"""
Phase 4 — Custom Gymnasium Environment
=======================================
AppianQueueEnv wraps the Phase 3 SimPy simulation engine as a
fully Gymnasium-compatible RL environment.

Observation space: Box(shape=(48,), dtype=float32)
    6 queues × 8 features:
    [wip_norm, utilization, breach_rate, avg_wait_norm,
     arrival_rate_norm, agents_norm, sla_norm, throughput_norm]

Action space: MultiDiscrete([5, 5, 5, 5, 5, 5])
    One delta level per queue: index → DELTAS[-2,-1,0,+1,+2]

Episode:
    - reset()   → randomize arrival noise, return obs
    - step()    → apply action, run 10 sim-runs, compute reward, return obs
    - done after max_steps steps (default 20)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from phase3_simulation.simpy_engine import (
    QueueConfig,
    QueueSimulation,
    SimulationResult,
    DEFAULT_QUEUE_CONFIGS,
    COMPLEXITY_WEIGHTS,
    COMPLEXITY_MULTIPLIERS,
)
from phase4_rl.action_space import (
    ActionSpace,
    QUEUE_NAMES,
    QUEUE_DEFAULT_AGENTS,
    QUEUE_MIN_AGENTS,
    QUEUE_MAX_AGENTS,
)
from phase4_rl.reward_shaper import RewardShaper, RewardWeights

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Observation normalisation constants
# ─────────────────────────────────────────────────────────────────────────────
_OBS_MAX = {
    "wip":           60.0,    # cases in system
    "utilization":   1.0,
    "breach_rate":   1.0,
    "avg_wait":      8.0,     # hours
    "arrival_rate":  25.0,    # cases/hour
    "agents":        15.0,    # max agents
    "sla_hours":     48.0,    # longest SLA
    "throughput":    25.0,    # cases completed/hour
}

N_FEATURES_PER_QUEUE = 8
OBS_DIM = len(QUEUE_NAMES) * N_FEATURES_PER_QUEUE   # 48


# ─────────────────────────────────────────────────────────────────────────────
# Environment configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EnvConfig:
    horizon_hours:     float = 8.0     # simulation horizon per step
    sim_runs_per_step: int   = 10      # runs averaged per RL step (noise reduction)
    max_steps:         int   = 20      # episode length
    arrival_noise:     float = 0.20    # ±% randomisation on arrival rates
    base_seed:         int   = 42
    reward_weights:    RewardWeights = field(default_factory=RewardWeights)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
class AppianQueueEnv(gym.Env):
    """
    Custom Gymnasium environment for Appian queue staffing optimization.

    The agent observes current queue states and decides how many agents to
    add or remove from each queue. The SimPy simulator evaluates the outcome,
    and the reward shaper converts results into a scalar reward signal.
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.config      = config or EnvConfig()
        self.render_mode = render_mode
        self._shaper     = RewardShaper(self.config.reward_weights)

        # ── Spaces ────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low   = 0.0,
            high  = 1.0,
            shape = (OBS_DIM,),
            dtype = np.float32,
        )
        self.action_space = spaces.MultiDiscrete(ActionSpace.NVEC)

        # ── Internal state ────────────────────────────────────────────
        self._current_agents: Dict[str, int] = dict(QUEUE_DEFAULT_AGENTS)
        self._arrival_rates: Dict[str, float] = {}
        self._step_count: int = 0
        self._prev_breach_rate: float = 0.0
        self._rng = random.Random(self.config.base_seed)
        self._episode_seed: int = self.config.base_seed

        # Cached last results for render()
        self._last_results: List[SimulationResult] = []
        self._last_info: dict = {}

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to start of a new episode."""
        super().reset(seed=seed)
        if seed is not None:
            self._episode_seed = seed
            self._rng = random.Random(seed)

        # Randomise arrival rates ±arrival_noise for generalisation
        self._arrival_rates = self._sample_arrival_rates()
        self._current_agents = dict(QUEUE_DEFAULT_AGENTS)
        self._step_count = 0
        self._prev_breach_rate = 0.0
        self._shaper.reset_episode()
        self._last_results = []
        self._last_info = {}

        obs = self._get_obs()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Apply action, simulate, compute reward.

        Returns: (obs, reward, terminated, truncated, info)
        """
        # 1. Decode + apply action
        deltas = ActionSpace.decode(action)
        self._current_agents = ActionSpace.apply(deltas, self._current_agents)

        # 2. Build QueueConfigs with current agents + randomised arrival rates
        configs = self._build_configs()

        # 3. Run sim_runs_per_step independent simulation runs
        results = self._run_simulations(configs)
        self._last_results = results

        # 4. Compute reward
        reward, info = self._shaper.compute(results, action, self._prev_breach_rate)
        self._prev_breach_rate = info.get("mean_breach_rate", 0.0)
        self._last_info = info

        # 5. Advance step counter
        self._step_count += 1
        terminated = False
        truncated  = self._step_count >= self.config.max_steps

        # 6. Build observation
        obs = self._get_obs_from_results(results)

        info["step"] = self._step_count
        info["action_str"] = ActionSpace.action_to_str(action)
        info["current_agents"] = dict(self._current_agents)

        if truncated:
            info["episode_summary"] = self._shaper.episode_summary()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    def render(self) -> Optional[str]:
        """Render current queue state to terminal."""
        if not self._last_results:
            return None

        by_queue: Dict[str, List[SimulationResult]] = {}
        for r in self._last_results:
            by_queue.setdefault(r.queue_name, []).append(r)

        lines = [
            f"\n{'─'*62}",
            f"  Step {self._step_count:>3} | Action: {self._last_info.get('action_str','N/A')}",
            f"{'─'*62}",
            f"  {'Queue':<22} {'Agents':>6} {'Breach%':>8} {'Util%':>7} {'Tput/h':>7}",
            f"  {'─'*58}",
        ]
        for q in QUEUE_NAMES:
            agents = self._current_agents.get(q, 0)
            if q in by_queue:
                rs = by_queue[q]
                br   = np.mean([r.breach_rate      for r in rs]) * 100
                util = np.mean([r.utilization       for r in rs]) * 100
                tput = np.mean([r.throughput_per_hour for r in rs])
                lines.append(f"  {q:<22} {agents:>6} {br:>7.1f}% {util:>6.1f}% {tput:>7.1f}")

        lines.append(f"{'─'*62}")
        lines.append(
            f"  Reward: {self._last_info.get('reward_total', 0.0):+.3f}  |  "
            f"Breach: {self._last_info.get('mean_breach_rate', 0)*100:.1f}%  |  "
            f"Compliance: {self._last_info.get('mean_compliance', 0)*100:.1f}%"
        )
        output = "\n".join(lines)
        print(output)
        return output

    # ------------------------------------------------------------------
    def close(self):
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _sample_arrival_rates(self) -> Dict[str, float]:
        """Randomise arrival rates within ±arrival_noise of baseline."""
        noise = self.config.arrival_noise
        rates = {}
        for name, cfg in DEFAULT_QUEUE_CONFIGS.items():
            base = float(cfg["base_arrival"])
            factor = 1.0 + self._rng.uniform(-noise, noise)
            rates[name] = max(0.5, base * factor)
        return rates

    def _build_configs(self) -> Dict[str, QueueConfig]:
        """Build QueueConfig dict from current agents + sampled arrival rates."""
        configs = {}
        for name, cfg in DEFAULT_QUEUE_CONFIGS.items():
            configs[name] = QueueConfig(
                name                         = name,
                sla_hours                    = float(cfg["sla_hours"]),
                arrival_rate_per_hour        = self._arrival_rates.get(name, float(cfg["base_arrival"])),
                service_rate_per_agent_per_hour = 2.0,
                num_agents                   = self._current_agents.get(name, int(cfg["default_agents"])),
                complexity_weights           = list(COMPLEXITY_WEIGHTS),
                complexity_multipliers       = dict(COMPLEXITY_MULTIPLIERS),
            )
        return configs

    def _run_simulations(self, configs: Dict[str, QueueConfig]) -> List[SimulationResult]:
        """Run sim_runs_per_step simulations and return all results."""
        all_results: List[SimulationResult] = []
        for run_id in range(self.config.sim_runs_per_step):
            seed = self._episode_seed + self._step_count * 1000 + run_id
            for cfg in configs.values():
                sim = QueueSimulation(
                    config        = cfg,
                    run_id        = run_id,
                    seed          = seed,
                    horizon_hours = self.config.horizon_hours,
                    scenario_name = "rl_step",
                    record_timeline = False,
                )
                all_results.append(sim.run())
        return all_results

    def _get_obs(self) -> np.ndarray:
        """Return zero-initialized observation (pre-simulation)."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # Fill agent counts and arrival rates from current state
        for qi, q in enumerate(QUEUE_NAMES):
            base_offset = qi * N_FEATURES_PER_QUEUE
            cfg = DEFAULT_QUEUE_CONFIGS[q]
            obs[base_offset + 4] = self._arrival_rates.get(q, float(cfg["base_arrival"])) / _OBS_MAX["arrival_rate"]
            obs[base_offset + 5] = self._current_agents.get(q, int(cfg["default_agents"])) / _OBS_MAX["agents"]
            obs[base_offset + 6] = float(cfg["sla_hours"]) / _OBS_MAX["sla_hours"]
        return np.clip(obs, 0.0, 1.0)

    def _get_obs_from_results(self, results: List[SimulationResult]) -> np.ndarray:
        """Build 48-dim observation from simulation results."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        by_queue: Dict[str, List[SimulationResult]] = {}
        for r in results:
            by_queue.setdefault(r.queue_name, []).append(r)

        for qi, q in enumerate(QUEUE_NAMES):
            base = qi * N_FEATURES_PER_QUEUE
            cfg = DEFAULT_QUEUE_CONFIGS[q]
            if q in by_queue:
                rs = by_queue[q]
                obs[base + 0] = np.mean([r.peak_wip            for r in rs]) / _OBS_MAX["wip"]
                obs[base + 1] = np.mean([r.utilization          for r in rs]) / _OBS_MAX["utilization"]
                obs[base + 2] = np.mean([r.breach_rate          for r in rs]) / _OBS_MAX["breach_rate"]
                obs[base + 3] = np.mean([r.avg_wait_time_hours  for r in rs]) / _OBS_MAX["avg_wait"]
                obs[base + 7] = np.mean([r.throughput_per_hour  for r in rs]) / _OBS_MAX["throughput"]
            obs[base + 4] = self._arrival_rates.get(q, float(cfg["base_arrival"])) / _OBS_MAX["arrival_rate"]
            obs[base + 5] = self._current_agents.get(q, int(cfg["default_agents"])) / _OBS_MAX["agents"]
            obs[base + 6] = float(cfg["sla_hours"]) / _OBS_MAX["sla_hours"]

        return np.clip(obs, 0.0, 1.0).astype(np.float32)
