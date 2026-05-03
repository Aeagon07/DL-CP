"""
Phase 3 — Parallelised Monte Carlo Orchestrator
================================================
Runs N independent simulation replications using joblib multiprocessing
(loky backend — works on Windows without freeze_support).

Each run gets a unique seed = base_seed + run_id, ensuring full
reproducibility. One designated "viz run" records timeline frames
for the HTML animation.

Key classes:
  MonteCarloConfig  — runtime configuration
  MonteCarloEngine  — orchestrates runs across all scenarios
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from joblib import Parallel, delayed

from phase3_simulation.simpy_engine import (
    QueueConfig,
    QueueSimulation,
    SimulationResult,
)
from phase3_simulation.scenario_builder import Scenario

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MonteCarloConfig:
    n_runs:          int   = 1000     # number of replications per scenario
    horizon_hours:   float = 8.0     # simulation time window
    n_jobs:          int   = -1      # joblib workers (-1 = all cores)
    base_seed:       int   = 42      # reproducibility seed base
    ml_forecasts:    Optional[dict]  = None   # LSTM forecasts (if available)
    verbose_every:   int   = 100     # log progress every N scenarios


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker (picklable for joblib loky backend)
# ─────────────────────────────────────────────────────────────────────────────
def _run_single(
    run_id:           int,
    queue_configs:    Dict[str, QueueConfig],
    scenario_name:    str,
    horizon_hours:    float,
    base_seed:        int,
    viz_run_id:       int,
) -> List[SimulationResult]:
    """
    Execute one Monte Carlo run across ALL queues.
    Returns a list of SimulationResult — one per queue.
    This must be a module-level function (not a lambda or nested function)
    so that joblib/loky can pickle it on all platforms including Windows.
    """
    seed    = base_seed + run_id
    results = []

    for cfg in queue_configs.values():
        sim = QueueSimulation(
            config         = cfg,
            run_id         = run_id,
            seed           = seed,
            horizon_hours  = horizon_hours,
            scenario_name  = scenario_name,
            record_timeline= (run_id == viz_run_id),
        )
        results.append(sim.run())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Monte Carlo Engine
# ─────────────────────────────────────────────────────────────────────────────
class MonteCarloEngine:
    """
    Orchestrates Monte Carlo simulation across multiple scenarios.

    Usage:
        engine  = MonteCarloEngine(MonteCarloConfig(n_runs=1000))
        configs = ScenarioBuilder.baseline_configs()
        results = engine.run_all_scenarios(scenarios, configs)
        # results: { scenario_name: [SimulationResult, ...] }
    """

    def __init__(self, config: MonteCarloConfig):
        self.config = config

    # ------------------------------------------------------------------
    def run_scenario(
        self,
        scenario:      Scenario,
        base_configs:  Dict[str, QueueConfig],
    ) -> List[SimulationResult]:
        """
        Run N replications of one scenario across all queues.

        Returns a flat list of SimulationResult objects.
        Length = n_runs × n_queues.
        One run (the "viz run") will have timeline frames populated.
        """
        modified_configs = scenario.apply(base_configs)
        n_runs           = self.config.n_runs
        viz_run_id       = n_runs // 2   # median run used for animation

        logger.info(
            f"[{scenario.name}] Starting {n_runs} × {len(modified_configs)} queue runs "
            f"(horizon={self.config.horizon_hours}h, jobs={self.config.n_jobs})"
        )
        t0 = time.time()

        # Parallel execution — loky backend works on Windows without guard
        nested = Parallel(n_jobs=self.config.n_jobs, backend="loky", verbose=0)(
            delayed(_run_single)(
                run_id        = run_id,
                queue_configs = modified_configs,
                scenario_name = scenario.name,
                horizon_hours = self.config.horizon_hours,
                base_seed     = self.config.base_seed,
                viz_run_id    = viz_run_id,
            )
            for run_id in range(n_runs)
        )

        # Flatten: nested is List[List[SimulationResult]] → List[SimulationResult]
        flat: List[SimulationResult] = [r for run_results in nested for r in run_results]

        elapsed = time.time() - t0
        logger.info(
            f"[{scenario.name}] Completed {len(flat)} results in {elapsed:.1f}s"
        )
        return flat

    # ------------------------------------------------------------------
    def run_all_scenarios(
        self,
        scenarios:    List[Scenario],
        base_configs: Dict[str, QueueConfig],
    ) -> Dict[str, List[SimulationResult]]:
        """
        Run all scenarios sequentially (scenarios are run one-at-a-time;
        parallelism is within each scenario across runs).

        Returns:
            { scenario_name: [SimulationResult, ...] }
        """
        all_results: Dict[str, List[SimulationResult]] = {}

        for i, scenario in enumerate(scenarios, 1):
            logger.info(f"Scenario {i}/{len(scenarios)}: {scenario.name}")
            all_results[scenario.name] = self.run_scenario(scenario, base_configs)

        logger.info(
            f"All {len(scenarios)} scenarios complete. "
            f"Total results: {sum(len(v) for v in all_results.values())}"
        )
        return all_results

    # ------------------------------------------------------------------
    def extract_viz_timeline(
        self,
        results: List[SimulationResult],
    ) -> Dict[str, list]:
        """
        Extract the timeline frames from the designated viz run
        (run_id == n_runs // 2) for each queue.

        Returns:
            { queue_name: [TimelineFrame, ...] }
        """
        viz_id = self.config.n_runs // 2
        timeline: Dict[str, list] = {}

        for r in results:
            if r.run_id == viz_id and r.timeline:
                timeline[r.queue_name] = r.timeline

        return timeline
