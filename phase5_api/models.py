"""
Phase 5 — Pydantic Response Models
====================================
All API response schemas for the FastAPI dashboard endpoints.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from pydantic import BaseModel


class QueueStateModel(BaseModel):
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
    status: str          # "healthy" | "warning" | "critical"
    trend: str           # "improving" | "stable" | "degrading"
    arrivals_per_hour: float


class PredictionModel(BaseModel):
    queue_name: str
    breach_prob_1h: float
    breach_prob_4h: float
    breach_prob_8h: float
    confidence_lower: float
    confidence_upper: float
    horizon_series: List[Dict]   # [{t: minutes, prob: float}, ...]


class RecommendationActionModel(BaseModel):
    queue_name: str
    current_agents: int
    recommended_agents: int
    delta: int
    urgency: str   # "add" | "remove" | "hold"


class RecommendationModel(BaseModel):
    timestamp: str
    confidence: float
    predicted_breach_reduction: float
    narrative: str
    actions: List[RecommendationActionModel]
    overall_status: str


class MonteCarloModel(BaseModel):
    worst_case_breach_pct: float
    expected_breach_pct: float
    queues_at_risk: List[str]
    scenarios_ranked: List[Dict]
    recommendations: List[str]
    last_run_scenarios: int


class AnomalyModel(BaseModel):
    queue_name: str
    score: float          # 0-1
    severity: str         # "normal" | "low" | "medium" | "high"
    threshold: float
    history: List[float]  # last 30 data points


class SHAPFeatureModel(BaseModel):
    feature: str
    value: float
    shap_value: float
    direction: str   # "positive" | "negative"


class DashboardSnapshotModel(BaseModel):
    queues: List[QueueStateModel]
    predictions: List[PredictionModel]
    recommendation: RecommendationModel
    montecarlo: MonteCarloModel
    anomalies: List[AnomalyModel]
    shap_data: Dict[str, List[SHAPFeatureModel]]
    timestamp: str
    uptime_seconds: float
    total_simulations_run: int
