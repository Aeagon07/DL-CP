"""
Phase 3 — CLI Runner
=====================
Entry point that can run without Kafka, Redis, or Docker.
Uses only: simpy, joblib, numpy, scipy, and the phase3_simulation package.

Usage:
    python -m phase3_simulation.runner
    python -m phase3_simulation.runner --n-runs 200 --horizon 4 --scenarios baseline volume_spike_50pct
    python -m phase3_simulation.runner --list-scenarios
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase3.runner")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Phase 3 — Monte Carlo Simulation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test (100 runs, baseline only)
  python -m phase3_simulation.runner --n-runs 100 --scenarios baseline

  # Full 1000-run multi-scenario sweep
  python -m phase3_simulation.runner

  # Custom scenario subset
  python -m phase3_simulation.runner --scenarios baseline volume_spike_50pct combined_worst_case

  # List available scenarios
  python -m phase3_simulation.runner --list-scenarios
""",
    )
    p.add_argument("--n-runs",    type=int,   default=1000, help="Monte Carlo runs per scenario (default 1000)")
    p.add_argument("--horizon",   type=float, default=8.0,  help="Simulation horizon in hours (default 8.0)")
    p.add_argument("--n-jobs",    type=int,   default=-1,   help="Parallel workers (-1 = all cores)")
    p.add_argument("--seed",      type=int,   default=42,   help="Base random seed (default 42)")
    p.add_argument(
        "--scenarios", nargs="*", default=None,
        help="Scenario names to run. Omit to run all 10 built-in scenarios.",
    )
    p.add_argument("--output", type=str, default="phase3_simulation_report.html",
                   help="Output HTML report path")
    p.add_argument("--no-report",    action="store_true", help="Skip HTML report generation")
    p.add_argument("--no-animation", action="store_true", help="Skip timeline recording (faster)")
    p.add_argument("--list-scenarios", action="store_true", help="Print available scenarios and exit")
    p.add_argument("--open-browser",   action="store_true", help="Auto-open report in browser")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    # Lazy imports after arg parse so --list-scenarios is instant
    from phase3_simulation.scenario_builder import ScenarioBuilder
    from phase3_simulation.monte_carlo import MonteCarloEngine, MonteCarloConfig
    from phase3_simulation.risk_aggregator import RiskAggregator
    from phase3_simulation.report_generator import ReportGenerator, ReportConfig

    all_scenarios = ScenarioBuilder.get_all_scenarios()
    scenario_map = {s.name: s for s in all_scenarios}

    # ── List mode ───────────────────────────────────────────────────────────
    if args.list_scenarios:
        print("\nAvailable Phase 3 Scenarios:\n")
        for s in all_scenarios:
            print(f"  {s.name:<30}  {s.description}")
        print()
        return 0

    # ── Select scenarios ────────────────────────────────────────────────────
    if args.scenarios:
        missing = [n for n in args.scenarios if n not in scenario_map]
        if missing:
            logger.error(f"Unknown scenarios: {missing}. Use --list-scenarios to see available ones.")
            return 1
        selected = [scenario_map[n] for n in args.scenarios]
    else:
        selected = all_scenarios

    logger.info(f"Phase 3 Monte Carlo Simulation")
    logger.info(f"  Scenarios : {[s.name for s in selected]}")
    logger.info(f"  Runs/scen : {args.n_runs:,}")
    logger.info(f"  Horizon   : {args.horizon}h")
    logger.info(f"  Workers   : {args.n_jobs}")
    logger.info(f"  Seed      : {args.seed}")

    # ── Run simulations ─────────────────────────────────────────────────────
    cfg = MonteCarloConfig(
        n_runs        = args.n_runs,
        horizon_hours = args.horizon,
        n_jobs        = args.n_jobs,
        base_seed     = args.seed,
    )

    # Temporarily reduce n_runs to 1 for viz_run computation when --no-animation
    if args.no_animation:
        import phase3_simulation.monte_carlo as mc_mod
        original_viz = mc_mod._run_single
        # timeline recording already gated by run_id == viz_id; no patch needed

    engine       = MonteCarloEngine(cfg)
    base_configs = ScenarioBuilder.baseline_configs()

    t0 = time.time()
    all_results = engine.run_all_scenarios(selected, base_configs)
    elapsed = time.time() - t0

    total_sims = sum(len(v) for v in all_results.values())
    logger.info(f"Completed {total_sims:,} simulation results in {elapsed:.1f}s")

    # ── Risk aggregation ────────────────────────────────────────────────────
    agg = RiskAggregator()
    distributions = agg.compute_all_distributions(all_results)
    comparisons   = agg.compare_all_to_baseline(all_results)
    tail          = agg.tail_risk_summary(all_results)

    # ── Console summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  PHASE 3 RISK SUMMARY")
    print("═" * 60)
    print(f"  Worst-case P90 breach rate : {tail['worst_case_breach_rate']*100:.1f}%")
    print(f"  Expected breach rate       : {tail['expected_breach_rate']*100:.1f}%")
    at_risk = tail["queues_at_risk"]
    print(f"  Queues at risk (>15% P90)  : {', '.join(at_risk) if at_risk else 'None'}")
    print()
    print("  Scenario ranking (best → worst by P50 breach rate):")
    for rank, (name, p50) in enumerate(tail["scenarios_ranked"], 1):
        print(f"    {rank:2d}. {name:<30} {p50*100:.2f}%")
    print()
    for rec in tail.get("recommendations", []):
        print(f"  {rec}")
    print("═" * 60 + "\n")

    # ── HTML report ─────────────────────────────────────────────────────────
    if not args.no_report:
        rep_cfg = ReportConfig(
            output_path    = args.output,
            n_runs         = args.n_runs,
            horizon_hours  = args.horizon,
            include_animation = not args.no_animation,
            open_browser   = args.open_browser,
        )
        gen = ReportGenerator(rep_cfg)
        report_path = gen.generate_html_report(all_results, distributions, comparisons)
        logger.info(f"HTML report → {report_path}")
    else:
        logger.info("Report generation skipped (--no-report).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
