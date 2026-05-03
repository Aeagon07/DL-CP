# Appian Operations Center
## Predictive Process Simulation & Operational Forecasting
### 3rd Year Deep Learning Major Project — Production-Grade, Industry-Standard

---

## 🎯 Project Overview

An end-to-end intelligent system that transforms **reactive SLA monitoring** into **proactive AI-driven forecasting**. Instead of discovering breaches after they happen, operations managers can:

- See breach probabilities **1–4 hours before they happen** with SHAP explanations
- Run **Monte Carlo what-if experiments** virtually before committing changes
- Receive **RL-driven resource recommendations** in real-time
- Trust every forecast via **calibrated probabilities + conformal prediction intervals**
- Monitor **live operational data** streaming through Kafka on the dashboard
- Detect **anomalous operational patterns** via LSTM Autoencoder
- Understand **cross-queue cascade risk** via GNN dependency modelling

---

## 🏗️ Architecture

```
[Kafka + Zookeeper]  →  [Feature Consumer]  →  [Redis Feature Store]
      ↑                                               ↓
[Synthetic Producer]                    [Phase 2 ML Stack (20+ models)]
                                                      ↓
[PostgreSQL]  ←────────────────────────────  [FastAPI Backend]
[MLflow]           ←── all experiments         WebSocket Server
                                                      ↓
                                          [React Live Dashboard]
```

---

## 🔧 Prerequisites

| Tool | Purpose | Required When |
|------|---------|--------------|
| **Python 3.11** | Run all ML code | Always |
| **Docker Desktop** | Kafka, Redis, PostgreSQL, MLflow | Phase 1 full pipeline, training |
| **Node.js 18+** | React dashboard | Phase 5 only |

> **Quick note:** The 12 advanced DL smoke tests (`python phase2_ml/<module>.py`) run WITHOUT Docker — they use synthetic data. Docker is needed when you want the live Kafka stream → Redis → model pipeline.

---

## 🚀 Full Setup (One-Time)

### Step 1 — Install Python 3.11
```powershell
# Windows (PowerShell):
winget install Python.Python.3.11

# After install, close and reopen PowerShell to refresh PATH
python --version    # Should show Python 3.11.x
```

### Step 2 — Install ALL Python Dependencies (one command)
```powershell
cd "C:\DL CP"
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```
> `requirements.txt` covers **all 5 phases** — Kafka client, Redis, PyTorch, XGBoost, TFT, GNN, Federated Learning, FastAPI, SimPy, RL, MLflow, and more. One command installs everything.

### Step 3 — Configure Environment Variables
```powershell
copy .env.example .env
# Edit .env with your settings (defaults work for local dev)
```

### Step 4 — Start Infrastructure (Kafka + Redis + PostgreSQL + MLflow)
```powershell
docker-compose up -d
```

Services started by Docker:

| Service | URL / Port | What it does |
|---------|-----------|-------------|
| **Apache Kafka** | `localhost:9092` | Real-time event streaming |
| **Zookeeper** | `localhost:2181` | Kafka coordination |
| **Kafka UI** | http://localhost:8080 | Browse topics, messages |
| **Schema Registry** | http://localhost:8081 | Avro schema management |
| **Redis** | `localhost:6379` | Feature store + prediction cache |
| **PostgreSQL** | `localhost:5432` | Persistent storage |
| **MLflow** | http://localhost:5000 | Experiment tracking & model registry |

---

## 🧪 Run Smoke Tests (No Docker Needed)

Each advanced DL module can be tested standalone with synthetic data:

```powershell
# Activate venv first:
venv\Scripts\activate

# Run any module — all use synthetic data, no Kafka/Docker needed
python phase2_ml/conformal_prediction.py
python phase2_ml/lstm_autoencoder.py
python phase2_ml/multitask_learner.py
python phase2_ml/curriculum_learning.py
python phase2_ml/contrastive_pretrain.py
python phase2_ml/bayesian_nn.py
python phase2_ml/tft_forecaster.py
python phase2_ml/tab_transformer.py
python phase2_ml/gnn_dependency.py
python phase2_ml/neural_ode.py
python phase2_ml/maml_adaptor.py
python phase2_ml/federated_learning.py
```

Each script trains for 3–5 epochs and prints `✓ Module N — PASSED` when successful.

---

## 📡 Run Full Live Pipeline (Docker Required)

### Terminal 1 — Generate & Stream Live Events to Kafka
```powershell
venv\Scripts\activate
python data/synthetic_generator.py --mode stream --stream-interval 2
```

### Terminal 2 — Kafka Consumer → Feature Engineering → Redis
```powershell
venv\Scripts\activate
python phase1_pipeline/kafka_consumer.py --snapshot-interval 30
```

### Terminal 3 — Generate CSV Training Data
```powershell
venv\Scripts\activate
python data/synthetic_generator.py --mode csv --days 90
# Output: data/sample_data.csv (~7,000+ feature snapshots across 6 queues)
```

### Terminal 4 — Train All Models (logs to MLflow)
```powershell
venv\Scripts\activate
python phase2_ml/train.py --data data/sample_data.csv
# Duration: ~10–15 mins on CPU
# Track progress: http://localhost:5000
```

### Terminal 5 — Evaluate & Generate Report
```powershell
venv\Scripts\activate
python phase2_ml/evaluate.py --data data/sample_data.csv
# Target: AUC-ROC ≥ 0.88 | Brier Score ≤ 0.12
```

---

## 📁 Project Structure

```
C:\DL CP\
├── docker-compose.yml             ← Kafka + Zookeeper + Redis + PostgreSQL + MLflow
├── requirements.txt               ← ALL dependencies, all phases, one pip install
├── .env.example                   ← Template for environment variables
├── README.md
│
├── data/
│   ├── synthetic_generator.py     ← CSV dataset OR live Kafka stream mode
│   ├── sample_data.csv            ← Auto-generated (run step above)
│   └── raw_events.csv
│
├── scripts/
│   └── init_db.sql                ← PostgreSQL schema auto-init via Docker
│
├── phase1_pipeline/               ← LIVE DATA PIPELINE (requires Kafka+Redis)
│   ├── schemas.py                 ← Pydantic schemas for all Kafka messages
│   ├── kafka_producer.py          ← Idempotent producer, snappy compression
│   ├── kafka_consumer.py          ← Manual offset commit, rolling buffer
│   ├── feature_engineering.py     ← 5 core features + 30+ lag/rolling features
│   ├── feature_store.py           ← Redis CRUD: features, predictions, alerts
│   └── data_quality.py            ← Schema validation, dedup, outlier checks
│
├── phase2_ml/                     ← ML STACK (standalone + pipeline modes)
│   │
│   ├── ── Base Models ────────────────────────────────────────────────────────
│   ├── lstm_forecaster.py         ← BiLSTM + MC Dropout + multi-horizon heads
│   ├── xgboost_classifier.py      ← Walk-forward CV + BreachEnsemble
│   ├── calibration.py             ← Isotonic / Platt / Temperature scaling
│   ├── shap_explainer.py          ← SHAP TreeExplainer + human narratives
│   ├── drift_detection.py         ← Page-Hinkley + ADWIN + PSI monitor
│   ├── train.py                   ← Full orchestrated training pipeline
│   ├── evaluate.py                ← AUC-ROC, Brier, ECE, F1, confusion matrix
│   ├── mlflow_tracking.py         ← Experiment & model registry logging
│   │
│   ├── ── 12 Advanced DL Modules ─────────────────────────────────────────────
│   ├── conformal_prediction.py    ← Module 1:  Formally guaranteed CI (ICP)
│   ├── lstm_autoencoder.py        ← Module 2:  Unsupervised anomaly detection
│   ├── multitask_learner.py       ← Module 3:  Joint breach + volume heads
│   ├── curriculum_learning.py     ← Module 4:  Easy→hard sample scheduling
│   ├── contrastive_pretrain.py    ← Module 5:  SimCLR self-supervised (no labels)
│   ├── bayesian_nn.py             ← Module 6:  Bayes-by-Backprop BNN
│   ├── tft_forecaster.py          ← Module 7:  Temporal Fusion Transformer
│   ├── tab_transformer.py         ← Module 8:  Feature-level self-attention
│   ├── gnn_dependency.py          ← Module 9:  GAT cross-queue cascade graph
│   ├── neural_ode.py              ← Module 10: Continuous-time ODE dynamics
│   ├── maml_adaptor.py            ← Module 11: MAML few-shot (15 examples)
│   └── federated_learning.py      ← Module 12: FedAvg privacy-preserving
│
├── phase3_simulation/             ← MONTE CARLO WHAT-IF  (coming Phase 3)
├── phase4_rl/                     ← RL AUTO-OPTIMIZER      (coming Phase 4)
├── phase5_api/                    ← FASTAPI BACKEND        (coming Phase 5)
├── phase5_dashboard/              ← REACT LIVE DASHBOARD   (coming Phase 5)
│
├── models/                        ← Auto-created: saved .pt / .pkl artifacts
├── tests/                         ← Unit + integration tests
└── IMPROVEMENT_SUGGESTIONS.md     ← Original 12 DL concept backlog
```

---

## 🔬 Tech Stack

| Layer | Technology | Used In |
|-------|-----------|---------|
| **Streaming** | Apache Kafka (Confluent) + Docker | Phase 1 |
| **Cache** | Redis 7 | Phase 1 |
| **Database** | PostgreSQL 15 | Phase 1 |
| **ML Tracking** | MLflow | Phase 2 |
| **Deep Learning** | PyTorch 2.2 | Phase 2 (all 12 modules) |
| **Boosting** | XGBoost 2.0 + LightGBM | Phase 2 |
| **Explainability** | SHAP | Phase 2 |
| **Uncertainty** | Conformal Prediction + BNN | Phase 2 |
| **Graph Learning** | GAT (pure PyTorch) | Phase 2 |
| **Self-Supervised** | SimCLR NT-Xent | Phase 2 |
| **Meta-Learning** | MAML (FOMAML) | Phase 2 |
| **Federated** | FedAvg + flwr | Phase 2 |
| **Simulation** | SimPy 4 | Phase 3 |
| **RL** | Stable-Baselines3 PPO | Phase 4 |
| **API** | FastAPI + WebSocket | Phase 5 |
| **Frontend** | React + Chart.js | Phase 5 |
| **DevOps** | Docker Compose | All phases |

---

## 📊 Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| AUC-ROC | ≥ 0.88 | Per queue, per horizon |
| Brier Score | ≤ 0.12 | After isotonic calibration |
| Conformal Coverage | ≥ 80% | Guaranteed by construction |
| Monte Carlo | < 8 sec | 1,000 runs parallelised |
| RL Breach Rate | ≤ 60% baseline | PPO agent optimised |
| WebSocket Latency | < 1 sec | Live metrics push |

---

## 🧪 Run Tests
```powershell
venv\Scripts\activate
pytest tests/ -v --cov=.
```

---

## 📌 Quick Command Reference

```powershell
# ── Setup (run once) ──────────────────────────────────────
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# ── Infrastructure (requires Docker Desktop running) ──────
docker-compose up -d          # start Kafka, Redis, PostgreSQL, MLflow
docker-compose down           # stop all services
docker-compose logs -f kafka  # watch Kafka logs

# ── Data & Training ───────────────────────────────────────
python data/synthetic_generator.py --mode csv --days 90
python phase2_ml/train.py --data data/sample_data.csv
python phase2_ml/evaluate.py --data data/sample_data.csv

# ── Smoke test any DL module (no Docker needed) ───────────
python phase2_ml/tft_forecaster.py
python phase2_ml/gnn_dependency.py
python phase2_ml/federated_learning.py

# ── Live stream (3 terminals) ────────────────────────────
python data/synthetic_generator.py --mode stream --stream-interval 2
python phase1_pipeline/kafka_consumer.py --snapshot-interval 30
```

---

*Appian Operations Center | Deep Learning Major Project 2025–26*
*Architecture: Kafka → Redis → PyTorch (TFT/GNN/BNN/MAML/FedAvg) → FastAPI → React*
