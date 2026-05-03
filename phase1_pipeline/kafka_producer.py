"""
Phase 1 — Kafka Producer
=========================
Reads from the synthetic data generator and publishes events
to Kafka topics with full production config:
  - Idempotent producer (exactly-once semantics)
  - Snappy compression
  - Message-key routing by queue_name (partition affinity)
  - Structured JSON payloads with schema version
  - Graceful shutdown on SIGINT/SIGTERM
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from threading import Event
from typing import Any

import numpy as np
from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_EVENTS      = os.getenv("KAFKA_TOPIC_EVENTS",    "appian.events.raw")
TOPIC_FEATURES    = os.getenv("KAFKA_TOPIC_FEATURES",  "appian.features.computed")

SCHEMA_VERSION    = "1.0"

QUEUE_CONFIGS = {
    "Document Review":    {"sla_hours": 8,  "base_arrival": 12, "max_agents": 12},
    "Compliance Check":   {"sla_hours": 4,  "base_arrival": 8,  "max_agents": 8},
    "Payment Processing": {"sla_hours": 2,  "base_arrival": 20, "max_agents": 15},
    "Customer Onboarding":{"sla_hours": 24, "base_arrival": 6,  "max_agents": 10},
    "Risk Assessment":    {"sla_hours": 12, "base_arrival": 5,  "max_agents": 8},
    "Audit Preparation":  {"sla_hours": 48, "base_arrival": 3,  "max_agents": 6},
}


# ─────────────────────────────────────────────────────────────────────────────
# Topic Management
# ─────────────────────────────────────────────────────────────────────────────
def ensure_topics_exist(topics: list[str], num_partitions: int = 3, replication: int = 1):
    """Create Kafka topics if they do not already exist."""
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP_SERVERS})
    existing = admin.list_topics(timeout=10).topics.keys()
    to_create = [
        NewTopic(t, num_partitions=num_partitions, replication_factor=replication)
        for t in topics if t not in existing
    ]
    if to_create:
        fs = admin.create_topics(to_create)
        for topic, f in fs.items():
            try:
                f.result()
                logger.info(f"Created Kafka topic: {topic}")
            except Exception as e:
                logger.warning(f"Topic {topic} already exists or error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Producer Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_producer() -> Producer:
    """Create a production-grade Kafka producer."""
    return Producer({
        "bootstrap.servers":         BOOTSTRAP_SERVERS,
        "acks":                      "all",
        "enable.idempotence":        True,
        "retries":                   5,
        "max.in.flight.requests.per.connection": 5,
        "linger.ms":                 10,
        "batch.size":                65536,
        "compression.type":          "snappy",
        "socket.timeout.ms":         30000,
        "message.timeout.ms":        30000,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Seasonality Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _hour_factor(hour: int) -> float:
    factors = [0.05]*7 + [0.4, 0.4, 1.4, 1.4, 1.1, 1.1, 0.7, 1.2, 1.2, 1.0, 0.9, 0.9, 0.4, 0.4, 0.05, 0.05]
    return factors[hour] if hour < len(factors) else 0.05


def _day_factor(weekday: int) -> float:
    return {0: 1.3, 1: 1.0, 2: 1.0, 3: 1.1, 4: 1.4, 5: 0.3, 6: 0.05}.get(weekday, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Message Builders
# ─────────────────────────────────────────────────────────────────────────────
def build_event_message(queue_name: str, config: dict, now: datetime) -> dict[str, Any]:
    import uuid, random
    complexity = np.random.choice(["LOW", "MEDIUM", "HIGH"], p=[0.4, 0.35, 0.25])
    return {
        "_schema_version": SCHEMA_VERSION,
        "event_id":   str(uuid.uuid4()),
        "case_id":    str(uuid.uuid4())[:8].upper(),
        "queue_name": queue_name,
        "event_type": "ARRIVE",
        "timestamp":  now.isoformat() + "Z",
        "complexity":  complexity,
        "sla_deadline": (now + timedelta(hours=config["sla_hours"])).isoformat() + "Z",
        "source":      "kafka_producer_live",
    }


def build_feature_message(queue_name: str, config: dict, now: datetime, hour_factor: float) -> dict[str, Any]:
    wip = float(np.random.poisson(config["base_arrival"] * 2 * hour_factor))
    return {
        "_schema_version": SCHEMA_VERSION,
        "snapshot_time":   now.isoformat() + "Z",
        "queue_name":      queue_name,
        "wip_count":       wip,
        "throughput_15m":  float(max(0, np.random.lognormal(2.0, 0.3) * hour_factor)),
        "throughput_30m":  float(max(0, np.random.lognormal(2.1, 0.3) * hour_factor)),
        "throughput_60m":  float(max(0, np.random.lognormal(2.2, 0.3) * hour_factor)),
        "utilization_rate": float(min(1.0, max(0.0, np.random.beta(5, 2) * hour_factor))),
        "time_pressure":   float(min(1.0, max(0.0, np.random.beta(2, 5)))),
        "complexity_backlog": float(max(0, np.random.lognormal(3, 0.5) * hour_factor)),
        "sla_hours":       float(config["sla_hours"]),
        "hour_of_day":     float(now.hour),
        "day_of_week":     float(now.weekday()),
        "is_weekend":      float(int(now.weekday() >= 5)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Delivery Callbacks
# ─────────────────────────────────────────────────────────────────────────────
_delivered = 0
_failed = 0

def on_delivery(err, msg):
    global _delivered, _failed
    if err:
        _failed += 1
        logger.error(f"Delivery failed | topic={msg.topic()} key={msg.key()} error={err}")
    else:
        _delivered += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main Producer Loop
# ─────────────────────────────────────────────────────────────────────────────
class AppianProducer:
    def __init__(self, event_interval_sec: float = 2.0):
        self.event_interval = event_interval_sec
        self.stop_event = Event()
        self.producer = build_producer()
        self._seq = 0
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        logger.info(f"Shutdown signal received (signal={signum}), draining …")
        self.stop_event.set()

    def run(self):
        """Main loop — publishes events + features every `event_interval` seconds."""
        ensure_topics_exist([TOPIC_EVENTS, TOPIC_FEATURES])
        logger.info(f"Producer started | interval={self.event_interval}s | topics={TOPIC_EVENTS}, {TOPIC_FEATURES}")

        while not self.stop_event.is_set():
            now = datetime.utcnow()
            h_factor = _hour_factor(now.hour)
            d_factor = _day_factor(now.weekday())
            combined = h_factor * d_factor

            for queue_name, config in QUEUE_CONFIGS.items():
                # Arrivals
                lam = config["base_arrival"] * combined * (self.event_interval / 3600)
                num = max(0, int(np.random.poisson(max(0.01, lam))))
                for _ in range(num):
                    msg = build_event_message(queue_name, config, now)
                    msg["seq"] = self._seq
                    self._seq += 1
                    self.producer.produce(
                        TOPIC_EVENTS,
                        key=queue_name.encode(),
                        value=json.dumps(msg).encode(),
                        callback=on_delivery,
                    )

                # Feature snapshot
                feat = build_feature_message(queue_name, config, now, combined)
                self.producer.produce(
                    TOPIC_FEATURES,
                    key=queue_name.encode(),
                    value=json.dumps(feat).encode(),
                    callback=on_delivery,
                )

            self.producer.poll(0)

            if self._seq % 100 == 0:
                logger.info(
                    f"Published {_delivered:,} messages | failed={_failed} | seq={self._seq}"
                )

            time.sleep(self.event_interval)

        # Graceful flush
        remaining = self.producer.flush(timeout=15)
        logger.info(f"Producer stopped. Remaining={remaining} | delivered={_delivered:,} | failed={_failed}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Appian Kafka Producer")
    parser.add_argument("--interval", type=float, default=2.0, help="Publish interval in seconds")
    args = parser.parse_args()
    AppianProducer(event_interval_sec=args.interval).run()
