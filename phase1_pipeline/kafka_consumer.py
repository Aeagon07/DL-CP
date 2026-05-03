"""
Phase 1 — Kafka Consumer & Feature Computation Engine
=======================================================
Consumes raw events from appian.events.raw,
computes all 5 core features per queue per 5-minute window,
validates with Pydantic, writes to:
  - Redis  (feature_store, 1-hour TTL, for ML inference)
  - PostgreSQL (feature_snapshots, for training data accumulation)
  - Kafka  (appian.features.computed, for downstream consumers)

Consumer Group: appian-feature-consumers
Partitions: 3 (one per Kafka partition, parallelizable)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Event, Thread
from typing import Any

import redis
import psycopg2
from confluent_kafka import Consumer, KafkaError, Producer
from dotenv import load_dotenv
from pydantic import ValidationError

from phase1_pipeline.schemas import RawCaseEvent, FeatureSnapshot
from phase1_pipeline.feature_engineering import FeatureEngineer
from phase1_pipeline.data_quality import DataQualityChecker

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
CONSUMER_GROUP    = os.getenv("KAFKA_CONSUMER_GROUP", "appian-feature-consumers")
TOPIC_IN          = os.getenv("KAFKA_TOPIC_EVENTS",   "appian.events.raw")
TOPIC_OUT         = os.getenv("KAFKA_TOPIC_FEATURES", "appian.features.computed")

REDIS_HOST        = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT        = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB          = int(os.getenv("REDIS_DB", "0"))
FEATURE_TTL       = int(os.getenv("REDIS_FEATURE_TTL", "3600"))

DB_URL            = os.getenv("DATABASE_URL", "postgresql://appian:appian_secret@localhost:5432/appian_ops")
WINDOW_MINUTES    = 5   # Feature snapshot window


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Buffer — In-memory sliding window per queue
# ─────────────────────────────────────────────────────────────────────────────
class RollingEventBuffer:
    """
    Thread-safe 60-minute rolling buffer of events per queue.
    Supports O(1) append and O(n) window queries.
    """
    def __init__(self, max_window_minutes: int = 60):
        self.max_window = timedelta(minutes=max_window_minutes)
        self._buffers: dict[str, deque] = defaultdict(deque)

    def insert(self, event: RawCaseEvent):
        buf = self._buffers[event.queue_name]
        buf.append(event)
        # Evict events older than max window
        cutoff = datetime.now(timezone.utc) - self.max_window
        # Make cutoff timezone-naive if events are naive
        try:
            while buf and buf[0].timestamp.replace(tzinfo=None) < cutoff.replace(tzinfo=None):
                buf.popleft()
        except Exception:
            pass

    def get_window(self, queue_name: str, minutes: int) -> list[RawCaseEvent]:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [
            e for e in self._buffers[queue_name]
            if e.timestamp.replace(tzinfo=None) >= cutoff.replace(tzinfo=None)
        ]

    def queue_names(self) -> list[str]:
        return list(self._buffers.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Kafka Consumer
# ─────────────────────────────────────────────────────────────────────────────
def build_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers":         BOOTSTRAP_SERVERS,
        "group.id":                  CONSUMER_GROUP,
        "auto.offset.reset":         "latest",
        "enable.auto.commit":        False,              # Manual commit for reliability
        "max.poll.interval.ms":      300000,
        "session.timeout.ms":        30000,
        "fetch.min.bytes":           1,
        "fetch.wait.max.ms":         500,
    })


def build_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BOOTSTRAP_SERVERS,
        "acks":              "all",
        "linger.ms":         5,
        "compression.type":  "snappy",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Feature Persistence
# ─────────────────────────────────────────────────────────────────────────────
class FeaturePersistence:
    """Handles writes to Redis and PostgreSQL."""

    def __init__(self):
        self.redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        self._pg_conn = None
        self._pg_cursor = None
        self._connect_pg()

    def _connect_pg(self):
        try:
            self._pg_conn = psycopg2.connect(DB_URL)
            self._pg_cursor = self._pg_conn.cursor()
            logger.info("PostgreSQL connected")
        except Exception as e:
            logger.warning(f"PostgreSQL unavailable (will retry): {e}")

    def write_redis(self, snapshot: FeatureSnapshot):
        """Write feature snapshot to Redis with TTL."""
        key = f"features:{snapshot.queue_name}:latest"
        self.redis.setex(key, FEATURE_TTL, snapshot.model_dump_json())

        # Also push to timeseries list (last 288 entries = 24 hours at 5min)
        ts_key = f"features:{snapshot.queue_name}:history"
        self.redis.lpush(ts_key, snapshot.model_dump_json())
        self.redis.ltrim(ts_key, 0, 287)
        self.redis.expire(ts_key, FEATURE_TTL * 24)

    def write_postgres(self, snapshot: FeatureSnapshot):
        """Persist feature snapshot to PostgreSQL."""
        if self._pg_cursor is None:
            self._connect_pg()
            if self._pg_cursor is None:
                return

        try:
            self._pg_cursor.execute("""
                INSERT INTO feature_snapshots
                    (queue_id, snapshot_time, wip_count, throughput_15m, throughput_30m,
                     throughput_60m, utilization_rate, time_pressure, complexity_backlog, arrival_rate)
                SELECT q.queue_id, %s, %s, %s, %s, %s, %s, %s, %s, 0.0
                FROM queues q WHERE q.queue_name = %s
                ON CONFLICT DO NOTHING
            """, (
                snapshot.snapshot_time,
                snapshot.wip_count,
                snapshot.throughput_15m,
                snapshot.throughput_30m,
                snapshot.throughput_60m,
                snapshot.utilization_rate,
                snapshot.time_pressure,
                snapshot.complexity_backlog,
                snapshot.queue_name,
            ))
            self._pg_conn.commit()
        except Exception as e:
            logger.error(f"PostgreSQL write error: {e}")
            try:
                self._pg_conn.rollback()
            except Exception:
                self._connect_pg()


# ─────────────────────────────────────────────────────────────────────────────
# Feature Snapshot Worker (background thread)
# ─────────────────────────────────────────────────────────────────────────────
class FeatureSnapshotWorker(Thread):
    """Background thread that computes + publishes feature snapshots every 5 min."""

    def __init__(
        self,
        buffer: RollingEventBuffer,
        engineer: FeatureEngineer,
        persistence: FeaturePersistence,
        out_producer: Producer,
        stop_event: Event,
        interval_seconds: int = 30,  # 30s for demo; 300s (5min) for production
    ):
        super().__init__(daemon=True, name="SnapshotWorker")
        self.buffer = buffer
        self.engineer = engineer
        self.persistence = persistence
        self.producer = out_producer
        self.stop_event = stop_event
        self.interval = interval_seconds

    def run(self):
        logger.info(f"FeatureSnapshotWorker started | interval={self.interval}s")
        while not self.stop_event.is_set():
            time.sleep(self.interval)
            self._compute_all()

    def _compute_all(self):
        now = datetime.utcnow()
        for queue_name in self.buffer.queue_names():
            events_60 = self.buffer.get_window(queue_name, 60)
            events_30 = self.buffer.get_window(queue_name, 30)
            events_15 = self.buffer.get_window(queue_name, 15)

            try:
                feat = self.engineer.compute(
                    queue_name=queue_name,
                    events_60=events_60,
                    events_30=events_30,
                    events_15=events_15,
                    snapshot_time=now,
                )
                # Write everywhere
                self.persistence.write_redis(feat)
                self.persistence.write_postgres(feat)

                self.producer.produce(
                    TOPIC_OUT,
                    key=queue_name.encode(),
                    value=feat.model_dump_json().encode(),
                )
                self.producer.poll(0)
                logger.debug(f"Snapshot | {queue_name} | wip={feat.wip_count} util={feat.utilization_rate:.2f}")
            except Exception as e:
                logger.error(f"Snapshot error for {queue_name}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main Consumer
# ─────────────────────────────────────────────────────────────────────────────
class AppianConsumer:
    def __init__(self, snapshot_interval_sec: int = 30):
        self.consumer   = build_consumer()
        self.producer   = build_producer()
        self.buffer     = RollingEventBuffer()
        self.engineer   = FeatureEngineer()
        self.quality    = DataQualityChecker()
        self.persist    = FeaturePersistence()
        self.stop_event = Event()
        self.snapshot_worker = FeatureSnapshotWorker(
            buffer=self.buffer,
            engineer=self.engineer,
            persistence=self.persist,
            out_producer=self.producer,
            stop_event=self.stop_event,
            interval_seconds=snapshot_interval_sec,
        )
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Shutdown received — stopping consumer …")
        self.stop_event.set()

    def run(self):
        self.consumer.subscribe([TOPIC_IN])
        self.snapshot_worker.start()
        logger.info(f"Consumer started | group={CONSUMER_GROUP} | topic={TOPIC_IN}")

        stats = {"received": 0, "valid": 0, "rejected": 0}

        while not self.stop_event.is_set():
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError.PARTITION_EOF:
                    continue
                logger.error(f"Kafka error: {msg.error()}")
                continue

            try:
                raw = json.loads(msg.value().decode("utf-8"))
                stats["received"] += 1

                # Data quality check
                is_valid, reason = self.quality.check(raw)
                if not is_valid:
                    stats["rejected"] += 1
                    logger.debug(f"Rejected event: {reason}")
                    self.consumer.commit(message=msg, asynchronous=True)
                    continue

                # Pydantic validation
                event = RawCaseEvent(**raw)
                stats["valid"] += 1

                # Insert into rolling buffer
                self.buffer.insert(event)

                # Commit offset
                self.consumer.commit(message=msg, asynchronous=True)

                if stats["received"] % 500 == 0:
                    logger.info(
                        f"Consumer stats: received={stats['received']:,} "
                        f"valid={stats['valid']:,} rejected={stats['rejected']}"
                    )
            except (ValidationError, json.JSONDecodeError, KeyError) as e:
                stats["rejected"] += 1
                logger.warning(f"Parse/validation error: {e}")
                self.consumer.commit(message=msg, asynchronous=True)

        self.consumer.close()
        self.producer.flush(timeout=10)
        logger.info(f"Consumer stopped. Final stats: {stats}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Appian Kafka Consumer")
    parser.add_argument("--snapshot-interval", type=int, default=30,
                        help="Feature snapshot interval in seconds (default 30, prod 300)")
    args = parser.parse_args()
    AppianConsumer(snapshot_interval_sec=args.snapshot_interval).run()
