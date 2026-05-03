"""
Phase 1 — Redis Feature Store
==============================
Provides high-speed read/write access to feature snapshots and
prediction results for the inference pipeline.

Key patterns:
  features:{queue_name}:latest          → Latest FeatureSnapshot JSON
  features:{queue_name}:history         → List of last 288 snapshots (24h)
  predictions:{queue_name}:{horizon}    → Latest BreachPrediction JSON
  rl:recommendations:latest             → Latest RL recommendations
  alerts:active                         → Sorted set of active alerts (score = prob)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

import redis
from dotenv import load_dotenv

from phase1_pipeline.schemas import FeatureSnapshot, BreachPrediction, BreachAlert

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_HOST    = os.getenv("REDIS_HOST",       "localhost")
REDIS_PORT    = int(os.getenv("REDIS_PORT",   "6379"))
REDIS_DB      = int(os.getenv("REDIS_DB",     "0"))
FEATURE_TTL   = int(os.getenv("REDIS_FEATURE_TTL",   "3600"))   # 1 hour
SCENARIO_TTL  = int(os.getenv("REDIS_SCENARIO_TTL",  "86400"))  # 24 hours
HISTORY_LEN   = 288  # 24 hours at 5-min intervals


class FeatureStore:
    """
    Redis-backed feature store for real-time inference.
    All public methods are safe to call from multiple threads.
    """

    def __init__(self):
        self._r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        self._check_connection()

    def _check_connection(self):
        try:
            self._r.ping()
            logger.info(f"FeatureStore connected to Redis @ {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Snapshot CRUD
    # ─────────────────────────────────────────────────────────────────────────
    def set_features(self, snapshot: FeatureSnapshot) -> bool:
        """Write latest + push to history. Returns True on success."""
        try:
            pipe = self._r.pipeline()

            latest_key = f"features:{snapshot.queue_name}:latest"
            hist_key   = f"features:{snapshot.queue_name}:history"
            payload    = snapshot.model_dump_json()

            # Latest snapshot (with TTL)
            pipe.setex(latest_key, FEATURE_TTL, payload)

            # History ring-buffer (newest first)
            pipe.lpush(hist_key, payload)
            pipe.ltrim(hist_key, 0, HISTORY_LEN - 1)
            pipe.expire(hist_key, FEATURE_TTL * 24)

            pipe.execute()
            return True
        except Exception as e:
            logger.error(f"FeatureStore.set_features error: {e}")
            return False

    def get_latest_features(self, queue_name: str) -> Optional[FeatureSnapshot]:
        """Retrieve latest feature snapshot for a queue. Returns None if missing."""
        try:
            raw = self._r.get(f"features:{queue_name}:latest")
            if raw is None:
                return None
            return FeatureSnapshot.model_validate_json(raw)
        except Exception as e:
            logger.error(f"FeatureStore.get_latest_features error: {e}")
            return None

    def get_feature_history(
        self, queue_name: str, num_steps: int = 12
    ) -> list[FeatureSnapshot]:
        """
        Retrieve last `num_steps` feature snapshots (newest first).
        Used to construct LSTM input sequences.
        """
        try:
            raws = self._r.lrange(f"features:{queue_name}:history", 0, num_steps - 1)
            snapshots = []
            for raw in raws:
                try:
                    snapshots.append(FeatureSnapshot.model_validate_json(raw))
                except Exception:
                    continue
            return snapshots
        except Exception as e:
            logger.error(f"FeatureStore.get_feature_history error: {e}")
            return []

    def get_all_queues_latest(self) -> dict[str, FeatureSnapshot]:
        """Retrieve latest features for all known queues."""
        queues = [
            "Document Review", "Compliance Check", "Payment Processing",
            "Customer Onboarding", "Risk Assessment", "Audit Preparation",
        ]
        result = {}
        for q in queues:
            snap = self.get_latest_features(q)
            if snap:
                result[q] = snap
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction CRUD
    # ─────────────────────────────────────────────────────────────────────────
    def set_prediction(self, pred: BreachPrediction) -> bool:
        """Cache a breach prediction keyed by queue + horizon."""
        try:
            key = f"predictions:{pred.queue_name}:{pred.horizon_hours}"
            self._r.setex(key, FEATURE_TTL, pred.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"FeatureStore.set_prediction error: {e}")
            return False

    def get_prediction(
        self, queue_name: str, horizon_hours: int
    ) -> Optional[BreachPrediction]:
        """Get latest cached prediction for a queue + horizon."""
        try:
            raw = self._r.get(f"predictions:{queue_name}:{horizon_hours}")
            if raw is None:
                return None
            return BreachPrediction.model_validate_json(raw)
        except Exception as e:
            logger.error(f"FeatureStore.get_prediction error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Alert Store
    # ─────────────────────────────────────────────────────────────────────────
    def push_alert(self, alert: BreachAlert) -> bool:
        """Push alert into sorted set (score = breach probability)."""
        try:
            key = "alerts:active"
            self._r.zadd(key, {alert.model_dump_json(): alert.breach_prob})
            self._r.expire(key, 3600)
            # Keep only top 100 alerts
            self._r.zremrangebyrank(key, 0, -101)
            return True
        except Exception as e:
            logger.error(f"FeatureStore.push_alert error: {e}")
            return False

    def get_active_alerts(self, limit: int = 20) -> list[dict]:
        """Get top N alerts sorted by breach probability descending."""
        try:
            raw_list = self._r.zrevrange("alerts:active", 0, limit - 1)
            return [json.loads(r) for r in raw_list]
        except Exception as e:
            logger.error(f"FeatureStore.get_active_alerts error: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # RL Recommendations
    # ─────────────────────────────────────────────────────────────────────────
    def set_rl_recommendation(self, rec: dict) -> bool:
        try:
            self._r.setex("rl:recommendations:latest", FEATURE_TTL, json.dumps(rec))
            return True
        except Exception as e:
            logger.error(f"FeatureStore.set_rl_recommendation error: {e}")
            return False

    def get_rl_recommendation(self) -> Optional[dict]:
        try:
            raw = self._r.get("rl:recommendations:latest")
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error(f"FeatureStore.get_rl_recommendation error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Monte Carlo Scenario Cache
    # ─────────────────────────────────────────────────────────────────────────
    def set_scenario(self, scenario_hash: str, result: dict) -> bool:
        """Cache Monte Carlo scenario results by config hash."""
        try:
            key = f"scenario:{scenario_hash}"
            self._r.setex(key, SCENARIO_TTL, json.dumps(result))
            return True
        except Exception as e:
            logger.error(f"FeatureStore.set_scenario error: {e}")
            return False

    def get_scenario(self, scenario_hash: str) -> Optional[dict]:
        try:
            raw = self._r.get(f"scenario:{scenario_hash}")
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.error(f"FeatureStore.get_scenario error: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Health Check
    # ─────────────────────────────────────────────────────────────────────────
    def health(self) -> dict:
        try:
            latency_ms = self._r.execute_command("DEBUG SLEEP 0")
            info = self._r.info("memory")
            return {
                "status":      "ok",
                "used_memory": info.get("used_memory_human"),
                "connected":   True,
            }
        except Exception as e:
            return {"status": "error", "error": str(e), "connected": False}
