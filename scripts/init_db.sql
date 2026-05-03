-- =============================================================================
-- Appian Operations Center — Database Initialization
-- Runs automatically when PostgreSQL container starts for the first time
-- =============================================================================

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ── Queues / Agents Reference Data ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS queues (
    queue_id        SERIAL PRIMARY KEY,
    queue_name      VARCHAR(100) NOT NULL UNIQUE,
    queue_type      VARCHAR(50),          -- e.g. Review, Approval, Processing
    sla_hours       FLOAT NOT NULL DEFAULT 24.0,
    min_agents      INT NOT NULL DEFAULT 1,
    max_agents      INT NOT NULL DEFAULT 20,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agents (
    agent_id        SERIAL PRIMARY KEY,
    agent_name      VARCHAR(100) NOT NULL,
    queue_id        INT REFERENCES queues(queue_id),
    skill_level     FLOAT NOT NULL DEFAULT 1.0,
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Raw Events ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS case_events (
    event_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    case_id         VARCHAR(50) NOT NULL,
    queue_id        INT REFERENCES queues(queue_id),
    event_type      VARCHAR(50) NOT NULL,  -- ARRIVE, START, COMPLETE, ESCALATE
    timestamp       TIMESTAMPTZ NOT NULL,
    complexity      FLOAT,
    sla_deadline    TIMESTAMPTZ,
    assigned_agent  INT REFERENCES agents(agent_id),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_case_events_queue_time
    ON case_events(queue_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_case_events_case_id
    ON case_events(case_id);

-- ── Computed Features (snapshot every 5 min) ──────────────────────────────────
CREATE TABLE IF NOT EXISTS feature_snapshots (
    snapshot_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_id        INT REFERENCES queues(queue_id),
    snapshot_time   TIMESTAMPTZ NOT NULL,
    wip_count       INT NOT NULL,
    throughput_15m  FLOAT,
    throughput_30m  FLOAT,
    throughput_60m  FLOAT,
    utilization_rate FLOAT,
    time_pressure   FLOAT,
    complexity_backlog FLOAT,
    arrival_rate    FLOAT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feature_snapshots_queue_time
    ON feature_snapshots(queue_id, snapshot_time DESC);

-- ── Model Predictions ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_id        INT REFERENCES queues(queue_id),
    predicted_at    TIMESTAMPTZ NOT NULL,
    horizon_hours   INT NOT NULL,         -- 1, 2, or 4
    breach_probability FLOAT NOT NULL,
    breach_probability_calibrated FLOAT,
    volume_forecast FLOAT,
    shap_values     JSONB,               -- {feature: shap_value}
    model_version   VARCHAR(50),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_queue_horizon
    ON predictions(queue_id, horizon_hours, predicted_at DESC);

-- ── Alerts ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS alerts (
    alert_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_id        INT REFERENCES queues(queue_id),
    prediction_id   UUID REFERENCES predictions(prediction_id),
    alert_level     VARCHAR(20) NOT NULL,  -- LOW, MEDIUM, HIGH, CRITICAL
    breach_prob     FLOAT NOT NULL,
    horizon_hours   INT NOT NULL,
    message         TEXT,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    triggered_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_alerts_unack
    ON alerts(is_acknowledged, triggered_at DESC);

-- ── Drift Events ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS drift_events (
    drift_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    queue_id        INT REFERENCES queues(queue_id),
    detected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    drift_score     FLOAT NOT NULL,
    drift_type      VARCHAR(50),
    retraining_triggered BOOLEAN DEFAULT FALSE
);

-- ── Seed Reference Data ───────────────────────────────────────────────────────
INSERT INTO queues (queue_name, queue_type, sla_hours, min_agents, max_agents) VALUES
    ('Document Review',    'Review',     8.0,  3, 12),
    ('Compliance Check',   'Approval',   4.0,  2,  8),
    ('Payment Processing', 'Processing', 2.0,  4, 15),
    ('Customer Onboarding','Review',    24.0,  2, 10),
    ('Risk Assessment',    'Approval',  12.0,  2,  8),
    ('Audit Preparation',  'Review',    48.0,  1,  6)
ON CONFLICT (queue_name) DO NOTHING;
