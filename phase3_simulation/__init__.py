"""
Phase 3 — Monte Carlo What-If Simulation Engine
================================================
Top-level package. Exposes the primary public API.

Quick start:
    from phase3_simulation import MonteCarloEngine, ScenarioBuilder, RiskAggregator, ReportGenerator
    from phase3_simulation.monte_carlo import MonteCarloConfig

    scenarios = ScenarioBuilder.get_all_scenarios()
    engine    = MonteCarloEngine(MonteCarloConfig(n_runs=1000))
    results   = engine.run_all_scenarios(scenarios, ScenarioBuilder.baseline_configs())
    dists     = RiskAggregator().compute_all_distributions(results)
    comps     = RiskAggregator().compare_all_to_baseline(results)
    ReportGenerator().generate_html_report(results, dists, comps)
"""

from phase3_simulation.simpy_engine import (
    QueueConfig,
    QueueSimulation,
    SimulationResult,
    TimelineFrame,
    DEFAULT_QUEUE_CONFIGS,
)
from phase3_simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig
from phase3_simulation.scenario_builder import ScenarioBuilder, Scenario, ScenarioModification
from phase3_simulation.risk_aggregator import RiskAggregator, RiskDistribution, ScenarioComparison
from phase3_simulation.report_generator import ReportGenerator, ReportConfig

__all__ = [
    # Engine
    "QueueConfig", "QueueSimulation", "SimulationResult", "TimelineFrame",
    "DEFAULT_QUEUE_CONFIGS",
    # Monte Carlo
    "MonteCarloEngine", "MonteCarloConfig",
    # Scenarios
    "ScenarioBuilder", "Scenario", "ScenarioModification",
    # Risk
    "RiskAggregator", "RiskDistribution", "ScenarioComparison",
    # Report
    "ReportGenerator", "ReportConfig",
]
