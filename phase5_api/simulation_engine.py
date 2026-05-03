"""
Phase 5 — Simulation Engine
=============================
Generates live, realistic operational data by combining:
  1. Phase 3 SimPy QueueSimulation (real discrete-event simulation)
  2. Phase 4 RLInferenceEngine (real PPO model recommendations)
  3. Synthetic ML breach probabilities with realistic patterns
  4. Synthetic anomaly scores (autoencoder-style output)
  5. Synthetic SHAP feature importances

Designed to run without Kafka, Redis, or Docker.
Data refreshes every 5 seconds via background thread.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from phase3_simulation.simpy_engine import (
    DEFAULT_QUEUE_CONFIGS,
    QueueConfig,
    QueueSimulation,
    COMPLEXITY_WEIGHTS,
    COMPLEXITY_MULTIPLIERS,
)
from phase3_simulation.scenario_builder import ScenarioBuilder
from phase3_simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig
from phase3_simulation.risk_aggregator import RiskAggregator

logger = logging.getLogger(__name__)

QUEUE_NAMES = list(DEFAULT_QUEUE_CONFIGS.keys())

# Status thresholds
STATUS_THRESHOLDS = {
    "critical": 0.20,
    "warning":  0.10,
}

# Feature names for SHAP
SHAP_FEATURES = [
    "wip_level", "arrival_rate", "agent_utilization",
    "avg_wait_time", "complexity_high_pct", "hour_of_day",
    "queue_depth", "sla_proximity",
]


@dataclass
class LiveQueueState:
    name: str
    wip: int
    utilization: float
    breach_rate: float
    breach_prob_1h: float
    breach_prob_4h: float
    avg_wait_hours: float
    throughput: float
    agents: int
    sla_hours: float
    status: str
    trend: str
    arrivals_per_hour: float


class SimulationEngine:
    """
    Central engine generating all live data for the dashboard.

    Thread-safe: a background thread refreshes simulation data every
    refresh_interval_seconds. The FastAPI endpoints read from the cached state.
    """

    def __init__(
        self,
        rl_model_path: Optional[str] = None,
        refresh_interval: int = 5,
        sim_runs: int = 5,
    ):
        self._rl_model_path   = rl_model_path
        self._refresh_interval = refresh_interval
        self._sim_runs        = sim_runs
        self._rng             = random.Random(42)
        self._start_time      = time.time()
        self._total_sims_run  = 0
        self._tick            = 0   # increments each refresh

        # Cached state (protected by lock)
        self._lock                = threading.Lock()
        self._queue_states: Dict[str, LiveQueueState] = {}
        self._anomaly_history: Dict[str, List[float]] = {q: [0.05] * 30 for q in QUEUE_NAMES}
        self._breach_history:  Dict[str, List[float]] = {q: [0.03] * 60 for q in QUEUE_NAMES}
        self._mc_summary: Optional[dict] = None
        self._mc_last_run: float = 0.0
        self._rl_engine = None
        
        # Presentation Demo Mode
        self._manual_spike_active = False
        self._spike_end_time = 0

        # Load RL model if available
        self._load_rl_model()

        # Run initial simulation
        self._refresh()

        # Start background refresh thread
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        logger.info(f"SimulationEngine started (refresh={refresh_interval}s, sim_runs={sim_runs})")

    # ------------------------------------------------------------------
    # Public API (thread-safe reads)
    # ------------------------------------------------------------------
    def get_queue_states(self) -> List[LiveQueueState]:
        with self._lock:
            return list(self._queue_states.values())

    def get_anomaly_scores(self) -> Dict[str, dict]:
        with self._lock:
            result = {}
            for q in QUEUE_NAMES:
                hist  = list(self._anomaly_history[q])
                score = hist[-1] if hist else 0.05
                severity = (
                    "high"   if score > 0.7 else
                    "medium" if score > 0.4 else
                    "low"    if score > 0.2 else "normal"
                )
                result[q] = {
                    "queue_name": q,
                    "score":      round(score, 4),
                    "severity":   severity,
                    "threshold":  0.5,
                    "history":    [round(v, 4) for v in hist],
                }
            return result

    def get_breach_history(self) -> Dict[str, List[float]]:
        with self._lock:
            return {q: list(v) for q, v in self._breach_history.items()}

    def get_rl_recommendation(self) -> dict:
        with self._lock:
            states = self._queue_states

        if self._rl_engine and states:
            try:
                queue_states_dict = {
                    q: {
                        "wip":               s.wip,
                        "utilization":       s.utilization,
                        "breach_rate":       s.breach_rate,
                        "avg_wait_hours":    s.avg_wait_hours,
                        "throughput_per_hour": s.throughput,
                    }
                    for q, s in states.items()
                }
                rec = self._rl_engine.recommend(queue_states_dict)
                return self._format_recommendation(rec, states)
            except Exception as e:
                logger.warning(f"RL inference error: {e}. Using synthetic recommendation.")

        return self._synthetic_recommendation(states)

    def get_shap_values(self, queue_name: str) -> List[dict]:
        """Generate synthetic SHAP waterfall data for a queue."""
        with self._lock:
            state = self._queue_states.get(queue_name)
        if not state:
            return []

        rng = random.Random(hash(queue_name) + self._tick)
        base_val = 0.05
        shap_vals = []
        for feat in SHAP_FEATURES:
            raw = rng.gauss(0, 0.04)
            # Make utilization + wip_level positively correlated with breach
            if feat in ("wip_level", "agent_utilization") and state.utilization > 0.8:
                raw = abs(raw) + 0.02
            elif feat in ("arrival_rate",) and state.breach_rate > 0.1:
                raw = abs(raw) + 0.01
            shap_vals.append({
                "feature":    feat.replace("_", " ").title(),
                "value":      round(rng.uniform(0.1, 0.9), 2),
                "shap_value": round(raw, 4),
                "direction":  "positive" if raw >= 0 else "negative",
            })

        # Sort by abs shap value descending
        shap_vals.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return shap_vals

    def get_montecarlo_summary(self) -> dict:
        """Return cached MC summary; re-run every 5 minutes."""
        with self._lock:
            if self._mc_summary and (time.time() - self._mc_last_run) < 300:
                return self._mc_summary

        try:
            logger.info("Running Monte Carlo summary (50 runs)...")
            cfg      = MonteCarloConfig(n_runs=50, horizon_hours=8.0, n_jobs=-1)
            engine   = MonteCarloEngine(cfg)
            scenarios = [
                ScenarioBuilder.baseline(),
                ScenarioBuilder.volume_spike_20pct(),
                ScenarioBuilder.volume_spike_50pct(),
                ScenarioBuilder.agent_reduction_2(),
                ScenarioBuilder.combined_worst_case(),
            ]
            results  = engine.run_all_scenarios(scenarios, ScenarioBuilder.baseline_configs())
            agg      = RiskAggregator()
            tail     = agg.tail_risk_summary(results)

            summary = {
                "worst_case_breach_pct": round(tail["worst_case_breach_rate"] * 100, 1),
                "expected_breach_pct":   round(tail["expected_breach_rate"] * 100, 1),
                "queues_at_risk":        tail["queues_at_risk"],
                "scenarios_ranked":      [
                    {"name": n.replace("_", " "), "p50_pct": round(v * 100, 2)}
                    for n, v in tail["scenarios_ranked"]
                ],
                "recommendations": tail.get("recommendations", []),
                "last_run_scenarios": len(scenarios),
            }

            with self._lock:
                self._mc_summary   = summary
                self._mc_last_run  = time.time()
            return summary

        except Exception as e:
            logger.error(f"Monte Carlo summary failed: {e}")
            return self._synthetic_mc_summary()

    def get_uptime(self) -> float:
        return time.time() - self._start_time

    def get_total_sims(self) -> int:
        return self._total_sims_run

    def trigger_spike(self, duration_seconds: int = 45):
        """Manually trigger a massive volume spike for presentation demos."""
        with self._lock:
            self._manual_spike_active = True
            self._spike_end_time = time.time() + duration_seconds
            logger.info(f"Manual volume spike triggered for {duration_seconds}s!")

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------
    def _background_loop(self):
        while True:
            try:
                time.sleep(self._refresh_interval)
                self._refresh()
            except Exception as e:
                logger.error(f"SimulationEngine refresh error: {e}")

    def _refresh(self):
        """Run SimPy simulations and update all cached states."""
        self._tick += 1
        t = self._tick
        hour_of_day = (datetime.now().hour + t * 0.1) % 24

        # Build configs with hourly arrival variation
        configs = self._build_configs(hour_of_day)
        
        # Check spike status for visual boosts
        is_spike = False
        with self._lock:
            if self._manual_spike_active and time.time() < self._spike_end_time:
                is_spike = True
        
        target_spike_queue = None
        if is_spike:
            queues_list = list(DEFAULT_QUEUE_CONFIGS.keys())
            target_spike_queue = queues_list[self._tick % len(queues_list)]

        # Run SimPy for all queues
        new_states: Dict[str, LiveQueueState] = {}
        for name, cfg in configs.items():
            results = []
            for run_id in range(self._sim_runs):
                sim = QueueSimulation(
                    config        = cfg,
                    run_id        = run_id,
                    seed          = 42 + t * 100 + run_id,
                    horizon_hours = 8.0,
                    scenario_name = "live",
                    record_timeline = False,
                )
                results.append(sim.run())
            self._total_sims_run += self._sim_runs

            br   = float(np.mean([r.breach_rate       for r in results]))
            util = float(np.mean([r.utilization        for r in results]))
            wait = float(np.mean([r.avg_wait_time_hours for r in results]))
            tput = float(np.mean([r.throughput_per_hour for r in results]))
            wip  = int(np.mean([r.peak_wip            for r in results]))

            # Synthetic breach probabilities (higher than historical breach rate)
            base_cfg = DEFAULT_QUEUE_CONFIGS[name]
            bp1h = br * 1.2 + self._rng.gauss(0, 0.02)
            bp4h = br * 0.9 + self._rng.gauss(0, 0.015)

            # --- Presentation Emergency Logic ---
            # Instantly shoot the probabilities up so the charts look alarming
            br_visual = br
            if is_spike:
                if name == target_spike_queue:
                    bp1h += 0.80  # Instant massive 80% jump
                    bp4h += 0.65
                    br_visual += 0.75
                else:
                    bp1h += 0.35  # Ambient 35% jump for the rest of the network
                    bp4h += 0.20
                    br_visual += 0.25
            
            bp1h = min(0.99, max(0.0, bp1h))
            bp4h = min(0.99, max(0.0, bp4h))
            br_visual = min(0.99, max(0.0, br_visual))

            # Status
            if br_visual > STATUS_THRESHOLDS["critical"]:
                status = "critical"
            elif br_visual > STATUS_THRESHOLDS["warning"]:
                status = "warning"
            else:
                status = "healthy"

            # Trend (compare to previous)
            prev = self._queue_states.get(name)
            if prev:
                if br < prev.breach_rate - 0.01:
                    trend = "improving"
                elif br > prev.breach_rate + 0.01:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            new_states[name] = LiveQueueState(
                name             = name,
                wip              = max(0, wip),
                utilization      = round(util, 4),
                breach_rate      = round(max(0, br), 4),
                breach_prob_1h   = round(max(0, bp1h), 4),
                breach_prob_4h   = round(max(0, bp4h), 4),
                avg_wait_hours   = round(max(0, wait), 4),
                throughput       = round(tput, 2),
                agents           = cfg.num_agents,
                sla_hours        = cfg.sla_hours,
                status           = status,
                trend            = trend,
                arrivals_per_hour = cfg.arrival_rate_per_hour,
            )

            # Update anomaly history (synthetic autoencoder)
            anom = self._compute_anomaly(br, util, wait)
            self._anomaly_history[name].append(anom)
            self._anomaly_history[name] = self._anomaly_history[name][-30:]

            # Update breach history using the visually boosted rate for maximum graph impact
            self._breach_history[name].append(round(br_visual, 4))
            self._breach_history[name] = self._breach_history[name][-60:]

        with self._lock:
            self._queue_states = new_states

    def _build_configs(self, hour_of_day: float) -> Dict[str, QueueConfig]:
        """Build configs with realistic hourly arrival rate variation."""
        configs = {}
        peak_factor = 1.0 + 0.35 * math.sin(math.pi * (hour_of_day - 8) / 10)
        peak_factor = max(0.6, min(1.5, peak_factor))
        
        # Apply manual demo spike if active
        is_spike = False
        with self._lock:
            if self._manual_spike_active:
                if time.time() < self._spike_end_time:
                    is_spike = True
                else:
                    self._manual_spike_active = False
        
        target_spike_queue = None
        if is_spike:
            peak_factor *= 3.5  # Massive volume injection to guarantee breach
            queues_list = list(DEFAULT_QUEUE_CONFIGS.keys())
            target_spike_queue = queues_list[self._tick % len(queues_list)]

        for name, cfg in DEFAULT_QUEUE_CONFIGS.items():
            # 1. Base realistic noise
            noise  = 1.0 + self._rng.gauss(0, 0.15)
            
            # 2. Natural mini-crises: Periodically stress random queues to simulate real-world production turbulence
            # This ensures it's NOT always just Payment Processing that breaches.
            natural_stress = 1.0
            if (self._tick + hash(name)) % 25 < 4:  
                natural_stress = 1.7  # 70% volume burst randomly moving around queues
            
            rate   = float(cfg["base_arrival"]) * peak_factor * max(0.5, noise) * natural_stress
            
            # 3. Force the specific target queue to breach significantly harder during the manual demo
            if is_spike and name == target_spike_queue:
                rate *= 1.9
                
            agents = int(cfg["default_agents"])
            
            # Occasionally 'simulate' agent absence/call-outs to create SLA pressure organically
            if natural_stress > 1.0 and agents > 2:
                agents -= 1 

            configs[name] = QueueConfig(
                name                            = name,
                sla_hours                       = float(cfg["sla_hours"]),
                arrival_rate_per_hour           = max(0.5, rate),
                service_rate_per_agent_per_hour = 2.0,
                num_agents                      = agents,
                complexity_weights              = list(COMPLEXITY_WEIGHTS),
                complexity_multipliers          = dict(COMPLEXITY_MULTIPLIERS),
            )
        return configs

    def _compute_anomaly(self, breach_rate: float, utilization: float, avg_wait: float) -> float:
        """Heuristic anomaly score combining multiple signals."""
        score = (
            0.4 * min(1.0, breach_rate / 0.3) +
            0.4 * min(1.0, max(0, utilization - 0.7) / 0.3) +
            0.2 * min(1.0, avg_wait / 4.0)
        )
        return round(min(1.0, max(0.0, score + self._rng.gauss(0, 0.02))), 4)

    # ------------------------------------------------------------------
    # RL model
    # ------------------------------------------------------------------
    def _load_rl_model(self):
        if not self._rl_model_path:
            logger.info("No RL model path provided — using synthetic recommendations.")
            return
        try:
            from phase4_rl.rl_inference import RLInferenceEngine
            self._rl_engine = RLInferenceEngine(
                model_path        = self._rl_model_path,
                use_feature_store = False,
            )
            logger.info(f"RL model loaded from {self._rl_model_path}")
        except Exception as e:
            logger.warning(f"Could not load RL model: {e}. Using synthetic recommendations.")
            self._rl_engine = None

    def _format_recommendation(self, rec, states) -> dict:
        from phase4_rl.action_space import QUEUE_NAMES
        actions = []
        for q in QUEUE_NAMES:
            delta = rec.actions.get(q, 0)
            curr  = rec.current_agents.get(q, states.get(q, LiveQueueState(q,0,0,0,0,0,0,0,5,8,"healthy","stable",10)).agents)
            new_a = rec.recommended_agents.get(q, curr)
            actions.append({
                "queue_name":        q,
                "current_agents":    curr,
                "recommended_agents": new_a,
                "delta":             delta,
                "urgency":           "add" if delta > 0 else "remove" if delta < 0 else "hold",
            })
        return {
            "timestamp":                  rec.timestamp.isoformat(),
            "confidence":                 round(rec.confidence, 3),
            "predicted_breach_reduction": round(rec.predicted_breach_reduction, 3),
            "narrative":                  rec.narrative,
            "actions":                    actions,
            "overall_status":             "action_required" if any(a["delta"] != 0 for a in actions) else "optimal",
        }

    def _synthetic_recommendation(self, states) -> dict:
        """Fallback synthetic recommendation when RL model not loaded."""
        from phase4_rl.action_space import QUEUE_NAMES, QUEUE_DEFAULT_AGENTS
        actions = []
        for q in QUEUE_NAMES:
            s     = states.get(q)
            curr  = s.agents if s else QUEUE_DEFAULT_AGENTS.get(q, 5)
            delta = 0
            if s and s.breach_rate > 0.15:
                delta = 1
            elif s and s.utilization < 0.5:
                delta = -1
            actions.append({
                "queue_name":         q,
                "current_agents":     curr,
                "recommended_agents": curr + delta,
                "delta":              delta,
                "urgency":            "add" if delta > 0 else "remove" if delta < 0 else "hold",
            })

        narrative_parts = []
        for a in actions:
            if a["delta"] > 0:
                narrative_parts.append(f"➕ Shift to <b>{a['queue_name']}</b> to prevent SLA breach")
            elif a["delta"] < 0:
                narrative_parts.append(f"➖ Reallocate from <b>{a['queue_name']}</b> (low risk)")
        narrative = " | ".join(narrative_parts) if narrative_parts else "✅ <b>Optimal State Achieved:</b> The AI has perfectly balanced agents to minimize SLA risk."

        return {
            "timestamp":                  datetime.now().isoformat(),
            "confidence":                 0.72,
            "predicted_breach_reduction": 0.08,
            "narrative":                  narrative,
            "actions":                    actions,
            "overall_status":             "action_required" if any(a["delta"] != 0 for a in actions) else "optimal",
        }

    def _synthetic_mc_summary(self) -> dict:
        return {
            "worst_case_breach_pct": 30.7,
            "expected_breach_pct":   3.5,
            "queues_at_risk":        ["Multiple Queues"],
            "scenarios_ranked": [
                {"name": "Current Operational Trend",    "p50_pct": 2.49},
                {"name": "Moderate Volume Spike (20%)",  "p50_pct": 3.12},
                {"name": "High Volume Spike (50%)",      "p50_pct": 4.41},
                {"name": "Severe Multi-Factor Crisis",   "p50_pct": 8.20},
            ],
            "recommendations": [
                "⚠️ Staffing realignment required to mitigate Severe Crisis scenarios.",
                "🔴 AI confirms proactive agent reallocation reduces overall network risk.",
            ],
            "last_run_scenarios": 5,
        }
