"""
Phase 3 — Risk Aggregator
==========================
Converts raw Monte Carlo SimulationResult lists into:
  - P10/P50/P90 risk distributions per (queue, scenario, metric)
  - Mann-Whitney U significance tests between scenarios
  - Value-at-Risk style tail risk quantification
  - Auto-generated narrative risk statements
  - Summary tables ready for HTML report consumption

Key classes:
  RiskDistribution   — statistical summary for one (queue, scenario, metric)
  ScenarioComparison — pairwise difference + significance for one metric
  RiskAggregator     — orchestrates all computations
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from phase3_simulation.simpy_engine import SimulationResult

logger = logging.getLogger(__name__)

# Metrics extracted from SimulationResult
METRICS = [
    "breach_rate",
    "avg_wait_time_hours",
    "p95_wait_time_hours",
    "utilization",
    "peak_wip",
    "throughput_per_hour",
    "sla_compliance_rate",
]

METRIC_LABELS = {
    "breach_rate":          "Breach Rate",
    "avg_wait_time_hours":  "Avg Wait (h)",
    "p95_wait_time_hours":  "P95 Wait (h)",
    "utilization":          "Utilization",
    "peak_wip":             "Peak WIP",
    "throughput_per_hour":  "Throughput/h",
    "sla_compliance_rate":  "SLA Compliance",
}

# Higher-is-better (for delta direction interpretation)
HIGHER_IS_BETTER = {
    "breach_rate":          False,
    "avg_wait_time_hours":  False,
    "p95_wait_time_hours":  False,
    "utilization":          False,   # high utilization = risk of overflow
    "peak_wip":             False,
    "throughput_per_hour":  True,
    "sla_compliance_rate":  True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RiskDistribution:
    """Full statistical distribution for one (queue, scenario, metric) triple."""
    queue_name:    str
    scenario_name: str
    metric:        str
    metric_label:  str
    n_runs:        int
    mean:          float
    std:           float
    p10:           float
    p25:           float
    p50:           float
    p75:           float
    p90:           float
    p95:           float
    p99:           float
    min_val:       float
    max_val:       float
    ci95_lower:    float   # 95% CI for the mean (t-distribution)
    ci95_upper:    float


@dataclass
class ScenarioComparison:
    """
    Pairwise comparison of one metric between a baseline and a comparison scenario
    for one queue, using a two-sided Mann-Whitney U test (non-parametric).
    """
    baseline_scenario:   str
    comparison_scenario: str
    queue_name:          str
    metric:              str
    baseline_p50:        float
    comparison_p50:      float
    delta_absolute:      float         # comparison_p50 - baseline_p50
    delta_relative_pct:  float         # % change relative to baseline_p50
    p_value:             float
    is_significant:      bool          # p < 0.05
    risk_verdict:        str           # "BETTER" | "WORSE" | "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Aggregator
# ─────────────────────────────────────────────────────────────────────────────
class RiskAggregator:
    """
    Converts flat List[SimulationResult] into structured risk analytics.

    Usage:
        agg   = RiskAggregator()
        dists = agg.compute_all_distributions(all_results)
        comps = agg.compare_all_to_baseline(all_results, baseline_name="baseline")
    """

    # ------------------------------------------------------------------
    def compute_distributions(
        self,
        results:       List[SimulationResult],
        scenario_name: str,
    ) -> List[RiskDistribution]:
        """
        Compute RiskDistribution for every (queue, metric) combination
        from the flat results list of one scenario.
        """
        from itertools import groupby

        # Group by queue
        by_queue: Dict[str, List[SimulationResult]] = {}
        for r in results:
            by_queue.setdefault(r.queue_name, []).append(r)

        distributions = []
        for queue_name, queue_results in by_queue.items():
            for metric in METRICS:
                values = np.array([getattr(r, metric) for r in queue_results], dtype=float)
                if len(values) == 0:
                    continue

                dist = self._summarise(values, queue_name, scenario_name, metric)
                distributions.append(dist)

        return distributions

    # ------------------------------------------------------------------
    def compute_all_distributions(
        self,
        all_results: Dict[str, List[SimulationResult]],
    ) -> Dict[str, List[RiskDistribution]]:
        """Compute distributions for every scenario. Returns scenario → list."""
        return {
            name: self.compute_distributions(results, name)
            for name, results in all_results.items()
        }

    # ------------------------------------------------------------------
    def compare_scenarios(
        self,
        baseline_results:    List[SimulationResult],
        comparison_results:  List[SimulationResult],
        baseline_name:       str,
        comparison_name:     str,
        metrics:             Optional[List[str]] = None,
    ) -> List[ScenarioComparison]:
        """
        Pairwise Mann-Whitney U test for every (queue, metric) combination.
        Uses two-sided test; significant if p < 0.05.
        """
        metrics = metrics or ["breach_rate", "avg_wait_time_hours", "sla_compliance_rate"]

        # Build lookup: queue → list of metric values
        def group(results):
            g: Dict[str, List[SimulationResult]] = {}
            for r in results:
                g.setdefault(r.queue_name, []).append(r)
            return g

        base_groups = group(baseline_results)
        comp_groups = group(comparison_results)
        queues = sorted(set(base_groups) & set(comp_groups))

        comparisons = []
        for queue in queues:
            for metric in metrics:
                base_vals = np.array([getattr(r, metric) for r in base_groups[queue]], dtype=float)
                comp_vals = np.array([getattr(r, metric) for r in comp_groups[queue]], dtype=float)

                if len(base_vals) < 5 or len(comp_vals) < 5:
                    continue

                base_p50 = float(np.median(base_vals))
                comp_p50 = float(np.median(comp_vals))
                delta    = comp_p50 - base_p50
                rel_pct  = (delta / max(abs(base_p50), 1e-9)) * 100.0

                try:
                    stat, p_val = scipy_stats.mannwhitneyu(base_vals, comp_vals, alternative="two-sided")
                except Exception:
                    p_val = 1.0

                is_sig = bool(p_val < 0.05)
                higher_better = HIGHER_IS_BETTER.get(metric, False)

                if is_sig:
                    improved = (delta > 0 and higher_better) or (delta < 0 and not higher_better)
                    verdict  = "BETTER" if improved else "WORSE"
                else:
                    verdict = "NEUTRAL"

                comparisons.append(ScenarioComparison(
                    baseline_scenario   = baseline_name,
                    comparison_scenario = comparison_name,
                    queue_name          = queue,
                    metric              = metric,
                    baseline_p50        = round(base_p50, 6),
                    comparison_p50      = round(comp_p50, 6),
                    delta_absolute      = round(delta, 6),
                    delta_relative_pct  = round(rel_pct, 2),
                    p_value             = round(p_val, 6),
                    is_significant      = is_sig,
                    risk_verdict        = verdict,
                ))

        return comparisons

    # ------------------------------------------------------------------
    def compare_all_to_baseline(
        self,
        all_results:   Dict[str, List[SimulationResult]],
        baseline_name: str = "baseline",
    ) -> List[ScenarioComparison]:
        """Compare every non-baseline scenario to baseline."""
        if baseline_name not in all_results:
            logger.warning(f"Baseline scenario '{baseline_name}' not found in results.")
            return []

        base_results = all_results[baseline_name]
        comparisons  = []

        for name, results in all_results.items():
            if name == baseline_name:
                continue
            comps = self.compare_scenarios(base_results, results, baseline_name, name)
            comparisons.extend(comps)

        return comparisons

    # ------------------------------------------------------------------
    def tail_risk_summary(
        self,
        all_results:   Dict[str, List[SimulationResult]],
    ) -> dict:
        """
        High-level summary dict:
          - worst_case_breach_rate   : max P90 breach_rate across all scenarios/queues
          - expected_breach_rate     : mean breach_rate across all runs
          - queues_at_risk           : queues with P90 breach_rate > 15% in baseline
          - scenarios_ranked         : scenarios sorted by P50 breach_rate (all queues avg)
          - recommendations          : auto-generated text list
        """
        worst   = 0.0
        all_br  = []
        queues_at_risk = set()

        scenario_p50s: Dict[str, float] = {}

        for scenario_name, results in all_results.items():
            by_queue: Dict[str, List[float]] = {}
            for r in results:
                by_queue.setdefault(r.queue_name, []).append(r.breach_rate)
                all_br.append(r.breach_rate)

            scenario_brs = []
            for queue, brs in by_queue.items():
                arr = np.array(brs)
                p90 = float(np.percentile(arr, 90))
                if p90 > worst:
                    worst = p90
                if scenario_name == "baseline" and p90 > 0.15:
                    queues_at_risk.add(queue)
                scenario_brs.append(float(np.median(arr)))

            scenario_p50s[scenario_name] = float(np.mean(scenario_brs)) if scenario_brs else 0.0

        ranked = sorted(scenario_p50s.items(), key=lambda x: x[1])

        return {
            "worst_case_breach_rate": round(worst, 4),
            "expected_breach_rate":   round(float(np.mean(all_br)) if all_br else 0, 4),
            "queues_at_risk":         sorted(queues_at_risk),
            "scenarios_ranked":       ranked,
            "recommendations":        self._auto_recommendations(queues_at_risk, ranked, scenario_p50s),
        }

    # ------------------------------------------------------------------
    def compute_var(
        self,
        results:    List[SimulationResult],
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Value-at-Risk style: breach_rate at the given confidence percentile per queue.
        Answers: "In the worst X% of simulation runs, breach rate exceeds Y."
        """
        by_queue: Dict[str, List[float]] = {}
        for r in results:
            by_queue.setdefault(r.queue_name, []).append(r.breach_rate)
        return {
            q: round(float(np.percentile(brs, confidence * 100)), 4)
            for q, brs in by_queue.items()
        }

    # ------------------------------------------------------------------
    def build_summary_table(
        self,
        all_results: Dict[str, List[SimulationResult]],
    ) -> List[dict]:
        """
        Build a list of dicts for easy tabular rendering in the HTML report.
        Each row: { scenario, queue, breach_p10, breach_p50, breach_p90, utilization_p50 }
        """
        rows = []
        for scenario_name, results in all_results.items():
            by_queue: Dict[str, List[SimulationResult]] = {}
            for r in results:
                by_queue.setdefault(r.queue_name, []).append(r)

            for queue, qresults in sorted(by_queue.items()):
                brs  = np.array([r.breach_rate for r in qresults])
                util = np.array([r.utilization for r in qresults])
                rows.append({
                    "scenario":        scenario_name,
                    "queue":           queue,
                    "breach_p10":      round(float(np.percentile(brs, 10)) * 100, 1),
                    "breach_p50":      round(float(np.percentile(brs, 50)) * 100, 1),
                    "breach_p90":      round(float(np.percentile(brs, 90)) * 100, 1),
                    "utilization_p50": round(float(np.percentile(util, 50)) * 100, 1),
                    "throughput_p50":  round(float(np.percentile(
                        [r.throughput_per_hour for r in qresults], 50)), 1),
                })
        return rows

    # ------------------------------------------------------------------
    @staticmethod
    def _summarise(
        values:        np.ndarray,
        queue_name:    str,
        scenario_name: str,
        metric:        str,
    ) -> RiskDistribution:
        n     = len(values)
        mean  = float(np.mean(values))
        std   = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se    = std / (n ** 0.5) if n > 1 else 0.0
        t_crit = 1.96   # approximate for large n

        return RiskDistribution(
            queue_name    = queue_name,
            scenario_name = scenario_name,
            metric        = metric,
            metric_label  = METRIC_LABELS.get(metric, metric),
            n_runs        = n,
            mean          = round(mean, 6),
            std           = round(std, 6),
            p10           = round(float(np.percentile(values, 10)),  6),
            p25           = round(float(np.percentile(values, 25)),  6),
            p50           = round(float(np.percentile(values, 50)),  6),
            p75           = round(float(np.percentile(values, 75)),  6),
            p90           = round(float(np.percentile(values, 90)),  6),
            p95           = round(float(np.percentile(values, 95)),  6),
            p99           = round(float(np.percentile(values, 99)),  6),
            min_val       = round(float(values.min()), 6),
            max_val       = round(float(values.max()), 6),
            ci95_lower    = round(mean - t_crit * se, 6),
            ci95_upper    = round(mean + t_crit * se, 6),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_recommendations(
        queues_at_risk: set,
        ranked:         List[Tuple[str, float]],
        scenario_p50s:  Dict[str, float],
    ) -> List[str]:
        recs = []
        if queues_at_risk:
            recs.append(
                f"⚠️  Queues at baseline risk (>15% P90 breach): "
                f"{', '.join(sorted(queues_at_risk))}. "
                f"Immediate staffing review recommended."
            )
        if len(ranked) > 1:
            worst_name, worst_p50 = ranked[-1]
            best_name,  best_p50  = ranked[0]
            recs.append(
                f"🔴 '{worst_name}' is the highest-risk scenario "
                f"(P50 breach rate: {worst_p50*100:.1f}%). "
                f"Avoid or add contingency staffing."
            )
            if best_name != "baseline":
                recs.append(
                    f"✅ '{best_name}' shows the best outcomes "
                    f"(P50 breach rate: {best_p50*100:.1f}%). "
                    f"Consider implementing its configuration changes."
                )
        return recs
