# Task Tracker — Predictive Process Simulation & Operational Forecasting
## Appian Operations Center — Deep Learning Major Project

---

## Phase 1 — Data Pipeline & Feature Engineering ✅ COMPLETE
- [x] `docker-compose.yml` — Kafka, Zookeeper, Redis, PostgreSQL, MLflow, Kafka-UI, Schema Registry
- [x] `requirements.txt` — All Python dependencies (updated with flwr, torchinfo)
- [x] `.env.example` — Environment configuration (all variables)
- [x] `README.md` — Full setup + run instructions
- [x] `scripts/init_db.sql` — PostgreSQL auto-init (tables, indexes, seed data)
- [x] `data/synthetic_generator.py` — Seasonality, CSV dataset, Kafka live stream
- [x] `phase1_pipeline/schemas.py` — Pydantic schemas (all messages)
- [x] `phase1_pipeline/kafka_producer.py` — Idempotent Kafka producer, snappy compression
- [x] `phase1_pipeline/kafka_consumer.py` — Manual offset commit, rolling buffer, snapshot worker
- [x] `phase1_pipeline/feature_engineering.py` — 5 core features + lag/rolling/derived features
- [x] `phase1_pipeline/feature_store.py` — Redis CRUD for features, predictions, alerts, RL recs
- [x] `phase1_pipeline/data_quality.py` — Schema, business rules, Redis dedup, outlier checks

---

## Phase 2 — ML Ensemble (Base + 12 Advanced DL Modules) ✅ COMPLETE

### Base ML Stack ✅
- [x] `phase2_ml/lstm_forecaster.py` — BiLSTM + multi-horizon heads + MC Dropout uncertainty
- [x] `phase2_ml/xgboost_classifier.py` — XGBoost + walk-forward CV + BreachEnsemble (all horizons)
- [x] `phase2_ml/calibration.py` — Isotonic + Platt + Temperature scaling + ECE/MCE diagnostics
- [x] `phase2_ml/shap_explainer.py` — TreeExplainer + narratives + waterfall data + LRU cache
- [x] `phase2_ml/drift_detection.py` — Page-Hinkley + ADWIN + PSI + DriftMonitor per queue
- [x] `phase2_ml/train.py` — Full pipeline orchestration (bug fix: undefined TRACKING_URI patched)
- [x] `phase2_ml/evaluate.py` — AUC-ROC, Brier, ECE, F1, confusion matrix, SHAP stats, report
- [x] `phase2_ml/mlflow_tracking.py` — Experiment logging for LSTM, XGBoost, drift, evaluation

### 🔴 HIGH IMPACT — Advanced DL Modules ✅
- [x] `phase2_ml/conformal_prediction.py` — **Module 1** | Split conformal prediction (ICP) — formally guaranteed 80% coverage intervals for LSTM + XGBoost outputs. ConformalRegressor, ConformalClassifier, CoverageReport.
- [x] `phase2_ml/lstm_autoencoder.py` — **Module 2** | Encoder-Decoder LSTM unsupervised anomaly detection. Trained on normal ops only. 95th-percentile threshold. Severity scoring (normal/low/medium/high). Redis integration.
- [x] `phase2_ml/gnn_dependency.py` — **Module 9** | Graph Attention Network (GAT) — 6-queue directed graph, cross-queue cascade risk. Pure PyTorch (no PyTorch Geometric). Interpretable attention weights per queue edge.
- [x] `phase2_ml/tft_forecaster.py` — **Module 7** | Temporal Fusion Transformer from scratch — GRN, VSN, multi-head attention, quantile heads (q10/q50/q90). Won M5 competition architecture.

### 🟡 MEDIUM IMPACT — Advanced DL Modules ✅
- [x] `phase2_ml/multitask_learner.py` — **Module 3** | Shared BiLSTM + dual heads (breach classification + volume regression). Adaptive GradNorm loss balancing λ.
- [x] `phase2_ml/curriculum_learning.py` — **Module 4** | 3-phase difficulty scheduling (easy→medium→hard). DifficultyScorer, CurriculumSampler (custom PyTorch Sampler), CurriculumSchedule.
- [x] `phase2_ml/contrastive_pretrain.py` — **Module 5** | SimCLR-style self-supervised pre-training. NT-Xent loss, 4 time-series augmentations (jitter, warp, scale, slice). SSLFeatureExtractor for XGBoost fine-tuning.
- [x] `phase2_ml/bayesian_nn.py` — **Module 6** | Bayes-by-Backprop variational inference. BayesianLinear layer, ELBO loss, aleatoric vs epistemic uncertainty decomposition.
- [x] `phase2_ml/neural_ode.py` — **Module 10** | Neural ODE with RK4 integrator (pure PyTorch). ODEFunc (2-layer ELU MLP), handles irregular time intervals. Continuous trajectory prediction.
- [x] `phase2_ml/maml_adaptor.py` — **Module 11** | First-Order MAML meta-learning. Few-shot adaptation (15 examples → new queue). MetaTask, MAMLTrainer, FewShotEvaluator sample-efficiency curve.

### 🟢 NICE TO HAVE — Advanced DL Modules ✅
- [x] `phase2_ml/tab_transformer.py` — **Module 8** | Feature Tokenizer + Transformer. Per-feature learned embeddings, [CLS] token classification, interpretable attention over features. Ablation vs XGBoost.
- [x] `phase2_ml/federated_learning.py` — **Module 12** | FedAvg across 6 queue silos. FederatedClient (private data), FederatedServer (weighted aggregation), FederationOrchestrator (R rounds). Privacy guarantee documented.

---

## Bug Fixes Applied ✅
- [x] `phase2_ml/train.py` line 303 — Fixed undefined `TRACKING_URI` → `os.getenv('MLFLOW_TRACKING_URI', ...)`

---

## Phase 3 — Monte Carlo What-If Simulation ✅ COMPLETE
**Goal**: Quantify operational risk under uncertainty using parallelized simulation.

- [x] `phase3_simulation/__init__.py` — Public API exports
- [x] `phase3_simulation/simpy_engine.py` — SimPy M/G/c discrete-event simulation, Poisson arrivals, lognormal service, SLA breach tracking, timeline frames
- [x] `phase3_simulation/monte_carlo.py` — 1,000-run joblib-parallel orchestrator (loky backend, Windows-safe)
- [x] `phase3_simulation/scenario_builder.py` — 10 built-in what-if scenarios (agent cuts, volume spikes, SLA tightening, worst-case combos)
- [x] `phase3_simulation/risk_aggregator.py` — P10/P50/P90 distributions, Mann-Whitney U significance testing, VaR, narrative recommendations
- [x] `phase3_simulation/report_generator.py` — Self-contained HTML report with animated pipeline, Chart.js breach distribution, scenario heatmap
- [x] `phase3_simulation/runner.py` — CLI entry point (--n-runs, --scenarios, --list-scenarios, --no-report flags)

**Run command**:
```
python -m phase3_simulation.runner --n-runs 100 --scenarios baseline volume_spike_50pct
python -m phase3_simulation.runner   # full 1000-run 10-scenario sweep
```

---

## Phase 4 — Reinforcement Learning Auto-Optimizer 🔜 UPCOMING
**Goal**: Train an RL agent to automatically reallocate agents across queues to minimize SLA breaches.

**Key Features to Build**:
- `phase4_rl/queue_env.py` — Custom Gymnasium environment (state=feature snapshot, action=agent reallocation)
- `phase4_rl/ppo_agent.py` — Proximal Policy Optimization (stable-baselines3 PPO) with action masking
- `phase4_rl/action_space.py` — Discrete action space: move N agents from queue A to queue B
- `phase4_rl/reward_shaper.py` — Reward = −breach_count + throughput_bonus − penalty for extreme actions
- `phase4_rl/rl_inference.py` — Real-time recommendation engine (subscribes to Kafka, emits actions)
- `phase4_rl/curriculum_env.py` — Progressive difficulty curriculum for RL training stability

**Why it matters**: Goes beyond prediction → prescriptive AI. The agent learns optimal staffing policies that maximize throughput and minimize SLA breaches simultaneously.

**Estimated effort**: 4–5 days

---

## Phase 5 — FastAPI Backend + React Live Dashboard + WebSocket 🔜 UPCOMING
**Goal**: Production-grade real-time web application that streams live predictions to a monitoring dashboard.

**Key Features to Build**:
- `phase5_api/main.py` — FastAPI app with REST endpoints + WebSocket server
- `phase5_api/routers/predictions.py` — GET /predict/{queue}/{horizon} → breach probability
- `phase5_api/routers/anomalies.py` — GET /anomalies/{queue} → current anomaly scores
- `phase5_api/routers/recommendations.py` — GET /recommend → RL agent recommendations
- `phase5_api/websocket_streamer.py` — WebSocket: push live predictions every 5 min (Kafka-driven)
- `phase5_api/model_registry.py` — Lazy-loads all 20 trained models (LSTM, TFT, GNN, BNN, etc.)
- `dashboard/` — React frontend with:
  - Live queue status cards (WIP, utilization, breach prob)
  - Real-time TFT attention weight visualization
  - GNN queue dependency graph (interactive)
  - Anomaly score timeline (autoencoder output)
  - SHAP waterfall charts per queue
  - RL action recommendation panel
  - Conformal prediction interval bands on volume forecast chart

**Why it matters**: Ties all phases together into a deployable product that demonstrates end-to-end capability: from raw Kafka events → ML predictions → live dashboard.

**Estimated effort**: 5–7 days

---

## Summary

| Phase | Status | Files | Notes |
|-------|--------|-------|-------|
| Phase 1 | ✅ Complete | 12 files | Full Kafka pipeline |
| Phase 2 Base | ✅ Complete | 8 files | LSTM + XGBoost + MLflow |
| Phase 2 Advanced | ✅ Complete | 12 files | All 12 DL modules |
| Phase 3 | ✅ Complete | 7 files | SimPy + Monte Carlo + HTML report |
| Phase 4 | 🔜 Queued | ~6 files | PPO RL agent |
| Phase 5 | 🔜 Queued | ~10 files | FastAPI + React dashboard |

**Total target: ~53 production-quality Python files across 5 phases.**
