"""
Phase 5 — CLI Runner
=====================
Starts the FastAPI server with the Simulation Engine.

Usage:
    python -m phase5_api.runner
    python -m phase5_api.runner --port 8000
    python -m phase5_api.runner --rl-model phase4_rl/checkpoints/best_model/best_model
    python -m phase5_api.runner --no-rl --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("phase5.runner")


def main(argv=None):
    p = argparse.ArgumentParser(description="Phase 5 — Appian Operations Center Dashboard")
    p.add_argument("--port",     type=int, default=8000,  help="Server port (default 8000)")
    p.add_argument("--host",     type=str, default="127.0.0.1")
    p.add_argument("--rl-model", type=str,
                   default="phase4_rl/checkpoints/best_model/best_model",
                   help="Path to trained PPO model (without .zip)")
    p.add_argument("--no-rl",    action="store_true",  help="Skip RL model loading")
    p.add_argument("--sim-runs", type=int, default=5,   help="SimPy runs per refresh (default 5)")
    p.add_argument("--refresh",  type=int, default=5,   help="Data refresh interval seconds (default 5)")
    p.add_argument("--reload",   action="store_true",   help="Enable auto-reload for development")
    args = p.parse_args(argv)

    from phase5_api.simulation_engine import SimulationEngine
    from phase5_api.main import app, set_engine

    rl_model = None if args.no_rl else args.rl_model

    logger.info("=" * 60)
    logger.info("  Appian Operations Center — Phase 5 Dashboard")
    logger.info("=" * 60)
    logger.info(f"  URL       : http://{args.host}:{args.port}")
    logger.info(f"  RL Model  : {rl_model or 'disabled (synthetic mode)'}")
    logger.info(f"  Sim Runs  : {args.sim_runs} per refresh")
    logger.info(f"  Refresh   : every {args.refresh}s")
    logger.info("=" * 60)
    logger.info("  Starting simulation engine...")

    engine = SimulationEngine(
        rl_model_path    = rl_model,
        refresh_interval = args.refresh,
        sim_runs         = args.sim_runs,
    )
    set_engine(engine)

    logger.info(f"  ✅ Engine ready. Opening dashboard at http://{args.host}:{args.port}")
    logger.info("  Press Ctrl+C to stop.")

    import uvicorn
    uvicorn.run(
        app,
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        log_level = "warning",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
