"""
Phase 1 — Pydantic Data Schemas
================================
Strict schemas for all messages flowing through the system.
Rejects malformed records before they touch the feature store.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Optional, ClassVar
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────
class EventType(str, Enum):
    ARRIVE   = "ARRIVE"
    START    = "START"
    COMPLETE = "COMPLETE"
    ESCALATE = "ESCALATE"
    SLA_BREACH = "SLA_BREACH"


class Complexity(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class AlertLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class HorizonHours(int, Enum):
    ONE  = 1
    TWO  = 2
    FOUR = 4


# ─────────────────────────────────────────────────────────────────────────────
# Incoming Raw Event (from Kafka)
# ─────────────────────────────────────────────────────────────────────────────
class RawCaseEvent(BaseModel):
    """Schema for raw case events arriving from Kafka topic appian.events.raw"""
    model_config = {"str_strip_whitespace": True}

    event_id:    str        = Field(..., description="Unique event identifier")
    case_id:     str        = Field(..., min_length=4, max_length=50)
    queue_name:  str        = Field(..., min_length=2, max_length=100)
    event_type:  EventType
    timestamp:   datetime
    complexity:  Complexity = Complexity.MEDIUM
    sla_deadline: datetime
    assigned_agent: Optional[int] = None
    source:      str        = "live"
    seq:         Optional[int] = None

    @field_validator("timestamp", "sla_deadline", mode="before")
    @classmethod
    def parse_iso(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v

    @model_validator(mode="after")
    def sla_must_be_after_arrival(self) -> "RawCaseEvent":
        if self.sla_deadline <= self.timestamp:
            raise ValueError(
                f"sla_deadline ({self.sla_deadline}) must be after timestamp ({self.timestamp})"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Feature Snapshot (written to Redis + PostgreSQL)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureSnapshot(BaseModel):
    """Computed features for one queue at one snapshot time."""
    snapshot_time:     datetime
    queue_name:        str        = Field(..., min_length=2)

    # Core features
    wip_count:         float      = Field(..., ge=0)
    throughput_15m:    float      = Field(..., ge=0)
    throughput_30m:    float      = Field(..., ge=0)
    throughput_60m:    float      = Field(..., ge=0)
    utilization_rate:  float      = Field(..., ge=0.0, le=1.0)
    time_pressure:     float      = Field(..., ge=0.0, le=1.0)
    complexity_backlog: float     = Field(..., ge=0)

    # Context features
    sla_hours:         float      = Field(..., gt=0)
    hour_of_day:       float      = Field(..., ge=0, le=23)
    day_of_week:       float      = Field(..., ge=0, le=6)
    is_weekend:        float      = Field(..., ge=0, le=1)

    # Label (present in training data, absent in live)
    breach_label:      Optional[int] = Field(None, ge=0, le=1)

    @property
    def feature_vector(self) -> list[float]:
        """Returns ordered feature list for ML model input."""
        return [
            self.wip_count,
            self.throughput_15m,
            self.throughput_30m,
            self.throughput_60m,
            self.utilization_rate,
            self.time_pressure,
            self.complexity_backlog,
            self.sla_hours,
            self.hour_of_day,
            self.day_of_week,
            self.is_weekend,
        ]

    FEATURE_NAMES: ClassVar[list[str]] = [
        "wip_count", "throughput_15m", "throughput_30m", "throughput_60m",
        "utilization_rate", "time_pressure", "complexity_backlog",
        "sla_hours", "hour_of_day", "day_of_week", "is_weekend",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# ML Prediction Output
# ─────────────────────────────────────────────────────────────────────────────
class BreachPrediction(BaseModel):
    """Output from the ML ensemble for one queue + horizon."""
    prediction_id:     str       = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    queue_name:        str
    predicted_at:      datetime
    horizon_hours:     HorizonHours
    breach_probability: float    = Field(..., ge=0.0, le=1.0)
    breach_probability_calibrated: float = Field(..., ge=0.0, le=1.0)
    volume_forecast:   float     = Field(..., ge=0)
    confidence_lower:  float     = Field(..., ge=0.0, le=1.0)
    confidence_upper:  float     = Field(..., ge=0.0, le=1.0)
    shap_values:       dict[str, float] = Field(default_factory=dict)
    model_version:     str       = "unknown"
    alert_level:       AlertLevel = AlertLevel.LOW


# ─────────────────────────────────────────────────────────────────────────────
# Alert
# ─────────────────────────────────────────────────────────────────────────────
class BreachAlert(BaseModel):
    """Alert record for the dashboard and downstream consumers."""
    alert_id:      str       = Field(default_factory=lambda: str(__import__("uuid").uuid4()))
    queue_name:    str
    alert_level:   AlertLevel
    breach_prob:   float     = Field(..., ge=0.0, le=1.0)
    horizon_hours: int
    message:       str
    top_factors:   list[str] = Field(default_factory=list)
    triggered_at:  datetime  = Field(default_factory=datetime.utcnow)
    is_acknowledged: bool    = False


# ─────────────────────────────────────────────────────────────────────────────
# Quality Report (output of data_quality.py checks)
# ─────────────────────────────────────────────────────────────────────────────
class DataQualityReport(BaseModel):
    """Summary of data quality checks for a batch of events."""
    total_received:    int
    valid_count:       int
    rejected_count:    int
    rejection_reasons: dict[str, int] = Field(default_factory=dict)  # reason → count
    quality_score:     float          = Field(..., ge=0.0, le=1.0)
    checked_at:        datetime       = Field(default_factory=datetime.utcnow)

    @property
    def pass_rate(self) -> float:
        return self.valid_count / max(1, self.total_received)
