"""
Phase 3 — SimPy Discrete-Event Simulation Engine
=================================================
Models Appian queue dynamics as a realistic M/G/c queue:
  - Poisson case arrivals (configurable rate)
  - Lognormal service times stratified by case complexity
  - Agent capacity as simpy.Resource (blocking on capacity)
  - SLA deadline tracking → breach detection per case
  - Optional per-minute timeline recording for live UI animation

Key classes:
  QueueConfig       — per-queue simulation parameters
  SimulationResult  — output of one simulation run
  TimelineFrame     — per-minute state snapshot for animation
  QueueSimulation   — SimPy simulation for one queue, one run
"""

from __future__ import annotations

import math
import logging
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import simpy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Queue baseline configurations (mirrors synthetic_generator.QUEUE_CONFIGS)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_QUEUE_CONFIGS: dict[str, dict] = {
    "Document Review":     {"sla_hours": 8,  "base_arrival": 12, "default_agents": 7,  "min_agents": 3,  "max_agents": 12},
    "Compliance Check":    {"sla_hours": 4,  "base_arrival": 8,  "default_agents": 5,  "min_agents": 2,  "max_agents": 8},
    "Payment Processing":  {"sla_hours": 2,  "base_arrival": 20, "default_agents": 9,  "min_agents": 4,  "max_agents": 15},
    "Customer Onboarding": {"sla_hours": 24, "base_arrival": 6,  "default_agents": 6,  "min_agents": 2,  "max_agents": 10},
    "Risk Assessment":     {"sla_hours": 12, "base_arrival": 5,  "default_agents": 5,  "min_agents": 2,  "max_agents": 8},
    "Audit Preparation":   {"sla_hours": 48, "base_arrival": 3,  "default_agents": 3,  "min_agents": 1,  "max_agents": 6},
}

COMPLEXITY_KEYS         = ["LOW", "MEDIUM", "HIGH"]
COMPLEXITY_WEIGHTS      = [0.40,  0.35,    0.25]       # matching generator
COMPLEXITY_MULTIPLIERS  = {"LOW": 0.7, "MEDIUM": 1.0, "HIGH": 1.8}

# Log-normal service time: mean = 30 * complexity_mult minutes, sigma = 0.5
_SERVICE_MEAN_MINUTES   = 30.0
_SERVICE_LOG_SIGMA      = 0.5
_SERVICE_MIN_SECONDS    = 60.0    # floor: 1 minute
_SERVICE_MAX_SECONDS    = 7200.0  # cap:   2 hours (prevent extreme outliers)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class QueueConfig:
    """
    Simulation parameters for one Appian queue.
    Scenario modifications change these fields before running.
    """
    name: str
    sla_hours: float                            # SLA deadline in hours
    arrival_rate_per_hour: float                # λ — cases/hour (Poisson)
    service_rate_per_agent_per_hour: float      # μ (informational only; lognormal is used)
    num_agents: int                             # c — server count (Resource capacity)
    complexity_weights: List[float] = field(
        default_factory=lambda: list(COMPLEXITY_WEIGHTS)
    )
    complexity_multipliers: dict = field(
        default_factory=lambda: dict(COMPLEXITY_MULTIPLIERS)
    )


@dataclass
class TimelineFrame:
    """Per-minute snapshot of simulation state — used by the HTML animation."""
    t_minutes: float
    wip: int                  # cases currently in system (waiting + in service)
    in_service: int           # cases actively being processed
    queue_length: int         # cases waiting for an agent
    completed: int            # cumulative completed
    breached: int             # cumulative breached
    arrivals_this_step: int = 0
    completions_this_step: int = 0
    breaches_this_step: int = 0


@dataclass
class SimulationResult:
    """
    Complete output from one simulation run.
    `timeline` is populated only on the designated "viz run" to keep
    joblib memory usage bounded.
    """
    run_id: int
    queue_name: str
    scenario_name: str
    sim_horizon_hours: float
    total_arrivals: int
    total_completed: int
    total_breached: int
    breach_rate: float
    avg_wait_time_hours: float
    p95_wait_time_hours: float
    peak_wip: int
    utilization: float
    sla_compliance_rate: float
    throughput_per_hour: float
    timeline: List[TimelineFrame] = field(default_factory=list, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Core SimPy Simulation
# ─────────────────────────────────────────────────────────────────────────────
class QueueSimulation:
    """
    SimPy discrete-event simulation of one Appian queue for one run.

    Design:
      • Poisson arrivals via exponential inter-arrival times
      • Each case spawns an independent SimPy process
      • Agents modelled as a simpy.Resource(capacity=num_agents)
      • Service time: Lognormal(log(30*complexity_mult*60), 0.5) seconds
      • SLA breach: completion_time > arrival_time + sla_hours * 3600
      • Timeline: optional periodic sampler (every 60 sim-seconds)
    """

    def __init__(
        self,
        config: QueueConfig,
        run_id: int,
        seed: int,
        horizon_hours: float = 8.0,
        scenario_name: str = "baseline",
        record_timeline: bool = False,
    ):
        self.config          = config
        self.run_id          = run_id
        self.seed            = seed
        self.horizon_hours   = horizon_hours
        self.horizon_seconds = horizon_hours * 3600.0
        self.scenario_name   = scenario_name
        self.record_timeline = record_timeline

        # Counters (reset per run)
        self._arrivals      = 0
        self._completed     = 0
        self._breached      = 0
        self._current_wip   = 0
        self._peak_wip      = 0
        self._wait_times: List[float] = []         # hours
        self._total_busy_time = 0.0                # seconds

        # Timeline (optional)
        self._frames: List[TimelineFrame] = []
        self._prev_completed = 0
        self._prev_arrivals  = 0
        self._prev_breached  = 0

    # ------------------------------------------------------------------
    def run(self) -> SimulationResult:
        """Execute the simulation and return a fully populated SimulationResult."""
        rng = random.Random(self.seed)

        env    = simpy.Environment()
        agents = simpy.Resource(env, capacity=max(1, self.config.num_agents))

        env.process(self._arrival_process(env, agents, rng))
        if self.record_timeline:
            env.process(self._timeline_recorder(env, agents))

        env.run(until=self.horizon_seconds)
        return self._collect_results()

    # ------------------------------------------------------------------
    def _arrival_process(self, env: simpy.Environment, agents: simpy.Resource, rng: random.Random):
        """Generator: Poisson arrivals at λ = arrival_rate_per_hour."""
        rate_per_sec = self.config.arrival_rate_per_hour / 3600.0
        if rate_per_sec <= 0:
            return

        while True:
            inter_arrival = rng.expovariate(rate_per_sec)
            yield env.timeout(inter_arrival)
            if env.now >= self.horizon_seconds:
                return

            self._arrivals     += 1
            self._current_wip  += 1
            self._peak_wip      = max(self._peak_wip, self._current_wip)

            # Sample complexity once per case
            complexity = rng.choices(COMPLEXITY_KEYS, weights=self.config.complexity_weights, k=1)[0]
            sla_deadline = env.now + self.config.sla_hours * 3600.0

            env.process(self._case_process(env, agents, env.now, sla_deadline, complexity, rng))

    # ------------------------------------------------------------------
    def _case_process(
        self,
        env: simpy.Environment,
        agents: simpy.Resource,
        arrive_time: float,
        sla_deadline: float,
        complexity: str,
        rng: random.Random,
    ):
        """Generator: one case waits for an agent, gets serviced, checks breach."""
        with agents.request() as req:
            yield req                           # block until agent available

            wait_hours = (env.now - arrive_time) / 3600.0
            self._wait_times.append(wait_hours)

            # Lognormal service time
            mult         = self.config.complexity_multipliers.get(complexity, 1.0)
            mu           = math.log(_SERVICE_MEAN_MINUTES * mult * 60.0)
            service_secs = math.exp(rng.gauss(mu, _SERVICE_LOG_SIGMA))
            service_secs = max(_SERVICE_MIN_SECONDS, min(service_secs, _SERVICE_MAX_SECONDS))

            self._total_busy_time += service_secs
            yield env.timeout(service_secs)

        # Agent released — record outcome
        self._current_wip -= 1
        self._completed   += 1
        if env.now > sla_deadline:
            self._breached += 1

    # ------------------------------------------------------------------
    def _timeline_recorder(self, env: simpy.Environment, agents: simpy.Resource):
        """Generator: records queue state every 60 simulated seconds."""
        while env.now < self.horizon_seconds:
            in_service   = agents.count
            queue_length = len(agents.queue)

            self._frames.append(TimelineFrame(
                t_minutes          = round(env.now / 60.0, 1),
                wip                = self._current_wip,
                in_service         = in_service,
                queue_length       = queue_length,
                completed          = self._completed,
                breached           = self._breached,
                arrivals_this_step = self._arrivals  - self._prev_arrivals,
                completions_this_step = self._completed - self._prev_completed,
                breaches_this_step    = self._breached  - self._prev_breached,
            ))
            self._prev_arrivals  = self._arrivals
            self._prev_completed = self._completed
            self._prev_breached  = self._breached

            yield env.timeout(60.0)

    # ------------------------------------------------------------------
    def _collect_results(self) -> SimulationResult:
        n          = max(1, self._arrivals)
        breach_rate = self._breached / n

        if self._wait_times:
            avg_wait = float(np.mean(self._wait_times))
            p95_wait = float(np.percentile(self._wait_times, 95))
        else:
            avg_wait = p95_wait = 0.0

        capacity_seconds = self.horizon_seconds * max(1, self.config.num_agents)
        utilization      = min(1.0, self._total_busy_time / capacity_seconds)
        throughput       = self._completed / max(0.001, self.horizon_hours)

        return SimulationResult(
            run_id               = self.run_id,
            queue_name           = self.config.name,
            scenario_name        = self.scenario_name,
            sim_horizon_hours    = self.horizon_hours,
            total_arrivals       = self._arrivals,
            total_completed      = self._completed,
            total_breached       = self._breached,
            breach_rate          = round(breach_rate, 6),
            avg_wait_time_hours  = round(avg_wait, 4),
            p95_wait_time_hours  = round(p95_wait, 4),
            peak_wip             = self._peak_wip,
            utilization          = round(utilization, 4),
            sla_compliance_rate  = round(1.0 - breach_rate, 6),
            throughput_per_hour  = round(throughput, 2),
            timeline             = self._frames,
        )
