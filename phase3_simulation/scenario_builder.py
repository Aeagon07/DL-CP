"""
Phase 3 — What-If Scenario Builder
====================================
Defines parameterised modifications to queue configurations so that the
Monte Carlo engine can answer "what happens if X changes?"

Built-in scenario library (10 presets):
  baseline              — No change; current staffing & arrival rates
  agent_reduction_2     — Remove 2 agents from Payment Processing
  agent_reduction_all   — Remove 1 agent from every queue
  agent_add_2_payment   — Add 2 agents to Payment Processing
  volume_spike_20pct    — 20 % arrival rate surge across all queues
  volume_spike_50pct    — 50 % arrival rate (Black Friday / quarter-end)
  sla_tighten_compliance— Compliance Check SLA: 4 h → 2 h
  sla_tighten_payment   — Payment Processing SLA: 2 h → 1 h
  high_complexity_surge — Shift complexity mix: 25 % HIGH → 50 % HIGH
  combined_worst_case   — 30 % volume spike + −1 agent all queues + SLA tightened

Custom scenarios can be built via ScenarioBuilder.from_dict(spec).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Union

from phase3_simulation.simpy_engine import (
    QueueConfig,
    DEFAULT_QUEUE_CONFIGS,
    COMPLEXITY_WEIGHTS,
    COMPLEXITY_MULTIPLIERS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Modification primitives
# ─────────────────────────────────────────────────────────────────────────────
class ScenarioParameter(str, Enum):
    AGENT_COUNT       = "num_agents"
    ARRIVAL_RATE      = "arrival_rate_per_hour"
    SLA_HOURS         = "sla_hours"
    SERVICE_RATE      = "service_rate_per_agent_per_hour"
    COMPLEXITY_HIGH   = "complexity_high_fraction"   # fraction of HIGH complexity cases


@dataclass
class ScenarioModification:
    """
    A single modification applied to one (or all) queues.

    mode:
      'absolute'   — set the parameter to `value` directly
      'delta'      — add `value` to the current parameter (can be negative)
      'multiplier' — multiply the current parameter by `value`
    """
    queue_name: str                                        # queue name or "ALL"
    parameter:  ScenarioParameter
    mode:       Literal["absolute", "delta", "multiplier"]
    value:      Union[float, int]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Scenario:
    """
    A named what-if scenario consisting of one or more modifications.
    `apply()` returns a deep-copied, modified dict of QueueConfig objects.
    """
    name:          str
    description:   str
    modifications: List[ScenarioModification] = field(default_factory=list)
    color:         str = "#6c757d"   # chart color for this scenario

    # ------------------------------------------------------------------
    def apply(self, base_configs: Dict[str, QueueConfig]) -> Dict[str, QueueConfig]:
        """
        Returns a modified deep-copy of base_configs with all modifications applied.
        Original is never mutated.
        """
        configs = copy.deepcopy(base_configs)

        for mod in self.modifications:
            targets = (
                list(configs.values())
                if mod.queue_name == "ALL"
                else [configs[mod.queue_name]] if mod.queue_name in configs else []
            )

            for cfg in targets:
                self._apply_single(cfg, mod)

        # Validate: enforce minimum agents
        for cfg in configs.values():
            cfg.num_agents = max(1, cfg.num_agents)
            cfg.arrival_rate_per_hour = max(0.1, cfg.arrival_rate_per_hour)
            cfg.sla_hours = max(0.5, cfg.sla_hours)

        return configs

    @staticmethod
    def _apply_single(cfg: QueueConfig, mod: ScenarioModification):
        param = mod.parameter.value  # actual attribute name

        if mod.parameter == ScenarioParameter.COMPLEXITY_HIGH:
            # Special handling: shift complexity mix HIGH fraction
            new_high = mod.value
            new_high = max(0.0, min(1.0, new_high))
            remaining = 1.0 - new_high
            cfg.complexity_weights = [
                round(remaining * 0.533, 4),   # LOW ~53% of non-HIGH
                round(remaining * 0.467, 4),   # MEDIUM ~47% of non-HIGH
                round(new_high, 4),
            ]
            return

        if not hasattr(cfg, param):
            logger.warning(f"QueueConfig has no attribute '{param}' — skipping")
            return

        current = getattr(cfg, param)
        if mod.mode == "absolute":
            new_val = mod.value
        elif mod.mode == "delta":
            new_val = current + mod.value
        elif mod.mode == "multiplier":
            new_val = current * mod.value
        else:
            raise ValueError(f"Unknown modification mode: {mod.mode}")

        # Cast back to int if original was int
        if isinstance(current, int):
            new_val = int(round(new_val))

        setattr(cfg, param, new_val)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
class ScenarioBuilder:
    """
    Factory for what-if scenarios.

    Usage:
        configs   = ScenarioBuilder.baseline_configs()
        scenarios = ScenarioBuilder.get_all_scenarios()
        modified  = scenarios[1].apply(configs)
    """

    # ------------------------------------------------------------------
    @staticmethod
    def baseline_configs() -> Dict[str, QueueConfig]:
        """
        Build the baseline QueueConfig dict from DEFAULT_QUEUE_CONFIGS.
        arrival_rate_per_hour = base_arrival (cases/hour in generator config).
        service_rate = 2 cases/agent/hour (30 min mean service time).
        """
        configs = {}
        for name, cfg in DEFAULT_QUEUE_CONFIGS.items():
            configs[name] = QueueConfig(
                name                         = name,
                sla_hours                    = float(cfg["sla_hours"]),
                arrival_rate_per_hour        = float(cfg["base_arrival"]),
                service_rate_per_agent_per_hour = 2.0,
                num_agents                   = int(cfg["default_agents"]),
                complexity_weights           = list(COMPLEXITY_WEIGHTS),
                complexity_multipliers       = dict(COMPLEXITY_MULTIPLIERS),
            )
        return configs

    # ------------------------------------------------------------------
    @staticmethod
    def get_all_scenarios() -> List[Scenario]:
        """Return the full library of 10 built-in what-if scenarios."""
        return [
            ScenarioBuilder.baseline(),
            ScenarioBuilder.agent_reduction_2(),
            ScenarioBuilder.agent_reduction_all(),
            ScenarioBuilder.agent_add_2_payment(),
            ScenarioBuilder.volume_spike_20pct(),
            ScenarioBuilder.volume_spike_50pct(),
            ScenarioBuilder.sla_tighten_compliance(),
            ScenarioBuilder.sla_tighten_payment(),
            ScenarioBuilder.high_complexity_surge(),
            ScenarioBuilder.combined_worst_case(),
        ]

    # ------------------------------------------------------------------
    @staticmethod
    def baseline() -> Scenario:
        return Scenario(
            name        = "baseline",
            description = "Current staffing levels and arrival rates — no changes.",
            modifications = [],
            color       = "#4361ee",
        )

    @staticmethod
    def agent_reduction_2() -> Scenario:
        return Scenario(
            name        = "agent_reduction_2",
            description = "Remove 2 agents from Payment Processing. Tests resilience to downsizing.",
            modifications = [
                ScenarioModification("Payment Processing", ScenarioParameter.AGENT_COUNT, "delta", -2),
            ],
            color = "#f72585",
        )

    @staticmethod
    def agent_reduction_all() -> Scenario:
        return Scenario(
            name        = "agent_reduction_all",
            description = "Remove 1 agent from every queue simultaneously (budget cut scenario).",
            modifications = [
                ScenarioModification("ALL", ScenarioParameter.AGENT_COUNT, "delta", -1),
            ],
            color = "#e63946",
        )

    @staticmethod
    def agent_add_2_payment() -> Scenario:
        return Scenario(
            name        = "agent_add_2_payment",
            description = "Add 2 agents to Payment Processing. Tests cost vs benefit of upstaffing.",
            modifications = [
                ScenarioModification("Payment Processing", ScenarioParameter.AGENT_COUNT, "delta", +2),
            ],
            color = "#06d6a0",
        )

    @staticmethod
    def volume_spike_20pct() -> Scenario:
        return Scenario(
            name        = "volume_spike_20pct",
            description = "20% arrival rate increase across all queues (moderate surge).",
            modifications = [
                ScenarioModification("ALL", ScenarioParameter.ARRIVAL_RATE, "multiplier", 1.20),
            ],
            color = "#f4a261",
        )

    @staticmethod
    def volume_spike_50pct() -> Scenario:
        return Scenario(
            name        = "volume_spike_50pct",
            description = "50% arrival rate surge across all queues (Black Friday / quarter-end peak).",
            modifications = [
                ScenarioModification("ALL", ScenarioParameter.ARRIVAL_RATE, "multiplier", 1.50),
            ],
            color = "#e76f51",
        )

    @staticmethod
    def sla_tighten_compliance() -> Scenario:
        return Scenario(
            name        = "sla_tighten_compliance",
            description = "Compliance Check SLA tightened from 4h to 2h (regulatory change).",
            modifications = [
                ScenarioModification("Compliance Check", ScenarioParameter.SLA_HOURS, "absolute", 2.0),
            ],
            color = "#7209b7",
        )

    @staticmethod
    def sla_tighten_payment() -> Scenario:
        return Scenario(
            name        = "sla_tighten_payment",
            description = "Payment Processing SLA tightened from 2h to 1h (SLA renegotiation).",
            modifications = [
                ScenarioModification("Payment Processing", ScenarioParameter.SLA_HOURS, "absolute", 1.0),
            ],
            color = "#9d4edd",
        )

    @staticmethod
    def high_complexity_surge() -> Scenario:
        return Scenario(
            name        = "high_complexity_surge",
            description = "Complexity mix shifts: HIGH cases surge from 25% to 50% of volume.",
            modifications = [
                ScenarioModification("ALL", ScenarioParameter.COMPLEXITY_HIGH, "absolute", 0.50),
            ],
            color = "#fb8500",
        )

    @staticmethod
    def combined_worst_case() -> Scenario:
        return Scenario(
            name        = "combined_worst_case",
            description = (
                "Combined stress test: 30% volume spike + remove 1 agent from all queues "
                "+ Payment Processing SLA halved."
            ),
            modifications = [
                ScenarioModification("ALL",                ScenarioParameter.ARRIVAL_RATE, "multiplier", 1.30),
                ScenarioModification("ALL",                ScenarioParameter.AGENT_COUNT,  "delta",       -1),
                ScenarioModification("Payment Processing", ScenarioParameter.SLA_HOURS,    "absolute",     1.0),
            ],
            color = "#d62828",
        )

    # ------------------------------------------------------------------
    @staticmethod
    def from_dict(spec: dict) -> Scenario:
        """
        Build a custom scenario from a plain dict. Example:
            spec = {
                "name": "my_test",
                "description": "Test removing 3 agents from Risk Assessment",
                "modifications": [
                    {
                        "queue_name": "Risk Assessment",
                        "parameter": "num_agents",
                        "mode": "delta",
                        "value": -3
                    }
                ]
            }
        """
        mods = []
        for m in spec.get("modifications", []):
            mods.append(ScenarioModification(
                queue_name = m["queue_name"],
                parameter  = ScenarioParameter(m["parameter"]),
                mode       = m["mode"],
                value      = m["value"],
            ))
        return Scenario(
            name          = spec["name"],
            description   = spec.get("description", ""),
            modifications = mods,
            color         = spec.get("color", "#6c757d"),
        )
