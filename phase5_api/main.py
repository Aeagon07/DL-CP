"""
Phase 5 — FastAPI Application
================================
REST + WebSocket backend for the Appian Operations Center dashboard.

Endpoints:
    GET  /                      → dashboard.html
    GET  /api/queues            → All 6 live queue states
    GET  /api/predict/{queue}   → Breach probabilities
    GET  /api/recommend         → RL staffing recommendation
    GET  /api/montecarlo        → Monte Carlo risk summary
    GET  /api/anomalies         → Anomaly scores all queues
    GET  /api/shap/{queue}      → SHAP waterfall data
    GET  /api/health            → Health check
    WS   /ws                    → Live dashboard stream (2s interval)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from phase5_api.simulation_engine import SimulationEngine, QUEUE_NAMES

logger = logging.getLogger(__name__)

# ─── Global engine instance (set by runner.py before app starts) ──────────────
_engine: Optional[SimulationEngine] = None

def set_engine(engine: SimulationEngine):
    global _engine
    _engine = engine

def get_engine() -> SimulationEngine:
    if _engine is None:
        raise RuntimeError("SimulationEngine not initialized. Call set_engine() first.")
    return _engine


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Appian Operations Center",
    description = "Real-time predictive operations dashboard powered by AI/ML",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─── WebSocket connection manager ────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)
        logger.info(f"WS disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: str):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ─── Helper: build full snapshot ─────────────────────────────────────────────
def _build_snapshot() -> dict:
    eng    = get_engine()
    states = eng.get_queue_states()
    anom   = eng.get_anomaly_scores()
    rec    = eng.get_rl_recommendation()
    mc     = eng.get_montecarlo_summary()
    hist   = eng.get_breach_history()

    queues_data = []
    preds_data  = []
    shap_data   = {}

    for s in states:
        h = hist.get(s.name, [0.03] * 60)
        queues_data.append({
            "name":             s.name,
            "wip":              s.wip,
            "utilization":      round(s.utilization * 100, 1),
            "breach_rate":      round(s.breach_rate * 100, 1),
            "breach_prob_1h":   round(s.breach_prob_1h * 100, 1),
            "breach_prob_4h":   round(s.breach_prob_4h * 100, 1),
            "avg_wait_hours":   s.avg_wait_hours,
            "throughput":       s.throughput,
            "agents":           s.agents,
            "sla_hours":        s.sla_hours,
            "status":           s.status,
            "trend":            s.trend,
            "arrivals_per_hour": s.arrivals_per_hour,
            "breach_history":   [round(v * 100, 1) for v in h[-20:]],
        })
        preds_data.append({
            "queue_name":       s.name,
            "breach_prob_1h":   round(s.breach_prob_1h * 100, 1),
            "breach_prob_4h":   round(s.breach_prob_4h * 100, 1),
            "breach_prob_8h":   round(min(99, s.breach_prob_4h * 0.85 * 100), 1),
        })
        shap_data[s.name] = eng.get_shap_values(s.name)

    return {
        "queues":       queues_data,
        "predictions":  preds_data,
        "recommendation": rec,
        "montecarlo":   mc,
        "anomalies":    list(anom.values()),
        "shap_data":    shap_data,
        "timestamp":    datetime.now().isoformat(),
        "uptime_seconds": round(eng.get_uptime(), 1),
        "total_simulations_run": eng.get_total_sims(),
    }


# ─── REST Endpoints ───────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the dashboard HTML file."""
    html_path = Path(__file__).parent / "dashboard.html"
    if not html_path.exists():
        return HTMLResponse("<h1>dashboard.html not found</h1>", status_code=404)
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    eng = get_engine()
    return {
        "status":        "ok",
        "uptime_seconds": round(eng.get_uptime(), 1),
        "total_sims":    eng.get_total_sims(),
        "queues_active": len(QUEUE_NAMES),
        "timestamp":     datetime.now().isoformat(),
    }


@app.get("/api/queues")
async def get_queues():
    snap = _build_snapshot()
    return JSONResponse(content={"queues": snap["queues"]})


@app.get("/api/predict/{queue_name}")
async def get_prediction(queue_name: str):
    eng    = get_engine()
    states = {s.name: s for s in eng.get_queue_states()}
    state  = states.get(queue_name)
    if not state:
        return JSONResponse({"error": f"Queue '{queue_name}' not found"}, status_code=404)
    return {
        "queue_name":     state.name,
        "breach_prob_1h": round(state.breach_prob_1h * 100, 1),
        "breach_prob_4h": round(state.breach_prob_4h * 100, 1),
        "breach_prob_8h": round(min(99, state.breach_prob_4h * 0.85 * 100), 1),
    }


@app.get("/api/recommend")
async def get_recommendation():
    return JSONResponse(content=get_engine().get_rl_recommendation())


@app.get("/api/montecarlo")
async def get_montecarlo():
    return JSONResponse(content=get_engine().get_montecarlo_summary())


@app.get("/api/anomalies")
async def get_anomalies():
    return JSONResponse(content={"anomalies": list(get_engine().get_anomaly_scores().values())})


@app.get("/api/shap/{queue_name}")
async def get_shap(queue_name: str):
    return JSONResponse(content={"shap": get_engine().get_shap_values(queue_name)})


@app.post("/api/trigger_spike")
async def trigger_spike():
    """Manually trigger a spike for presentation purposes."""
    get_engine().trigger_spike(duration_seconds=45)
    return JSONResponse(content={"status": "Spike triggered successfully! Expect breaches within 10 seconds."})


@app.get("/api/snapshot")
async def get_snapshot():
    """Full dashboard data snapshot."""
    return JSONResponse(content=_build_snapshot())


# ─── WebSocket ────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            snapshot = _build_snapshot()
            await websocket.send_text(json.dumps(snapshot))
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)
