"""
Phase 4 — CLI Runner
=====================
Entry point for all Phase 4 RL operations.

Commands:
    demo    — Run 5 random-action episodes (no model needed, instant)
    train   — Train PPO agent (default 200k steps)
    eval    — Evaluate a saved model
    infer   — Generate a single staffing recommendation

Examples:
    python -m phase4_rl.runner demo
    python -m phase4_rl.runner train --timesteps 50000 --no-curriculum
    python -m phase4_rl.runner train --timesteps 200000
    python -m phase4_rl.runner eval --model phase4_rl/checkpoints/best_model
    python -m phase4_rl.runner infer --model phase4_rl/checkpoints/best_model
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4.runner")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 4 — Reinforcement Learning Auto-Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  demo    No model required. Runs random-action episodes to verify the env.
  train   Full PPO training run. Saves best model to checkpoints/.
  eval    Load a saved model and run evaluation episodes.
  infer   Load a saved model and print a staffing recommendation.
        """,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── demo ──────────────────────────────────────────────────────────
    demo_p = sub.add_parser("demo", help="Quick env smoke-test (no model needed)")
    demo_p.add_argument("--episodes",   type=int,   default=5,    help="Number of demo episodes")
    demo_p.add_argument("--sim-runs",   type=int,   default=3,    help="Sim runs per step (fast demo)")
    demo_p.add_argument("--render",     action="store_true",      help="Print queue states each step")

    # ── train ─────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train PPO agent")
    train_p.add_argument("--timesteps",     type=int,   default=200_000)
    train_p.add_argument("--lr",            type=float, default=3e-4)
    train_p.add_argument("--n-steps",       type=int,   default=512)
    train_p.add_argument("--batch-size",    type=int,   default=64)
    train_p.add_argument("--sim-runs",      type=int,   default=10)
    train_p.add_argument("--horizon",       type=float, default=8.0)
    train_p.add_argument("--no-curriculum", action="store_true")
    train_p.add_argument("--checkpoint-dir", default="phase4_rl/checkpoints")
    train_p.add_argument("--eval-episodes", type=int,   default=10)

    # ── eval ──────────────────────────────────────────────────────────
    eval_p = sub.add_parser("eval", help="Evaluate a saved model")
    eval_p.add_argument("--model",    required=True, help="Path to model (without .zip)")
    eval_p.add_argument("--episodes", type=int, default=20)
    eval_p.add_argument("--sim-runs", type=int, default=10)
    eval_p.add_argument("--render",   action="store_true")

    # ── infer ─────────────────────────────────────────────────────────
    infer_p = sub.add_parser("infer", help="Generate a staffing recommendation")
    infer_p.add_argument("--model",              required=True)
    infer_p.add_argument("--use-feature-store",  action="store_true")
    infer_p.add_argument("--loop",               action="store_true",
                         help="Run continuously every --interval seconds")
    infer_p.add_argument("--interval",           type=int, default=300)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Command handlers
# ─────────────────────────────────────────────────────────────────────────────
def cmd_demo(args) -> int:
    """Run random-action episodes to verify the environment works."""
    import numpy as np
    from phase4_rl.queue_env import AppianQueueEnv, EnvConfig
    from phase4_rl.action_space import ActionSpace

    logger.info(f"Phase 4 Demo — {args.episodes} episodes, {args.sim_runs} sim-runs/step")

    cfg = EnvConfig(
        sim_runs_per_step = args.sim_runs,
        max_steps         = 5,
        arrival_noise     = 0.10,
    )
    render_mode = "human" if args.render else None
    env = AppianQueueEnv(cfg, render_mode=render_mode)

    episode_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0.0
        step = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        logger.info(
            f"Episode {ep+1}/{args.episodes}: "
            f"reward={ep_reward:.3f}, "
            f"breach={info.get('mean_breach_rate',0)*100:.1f}%, "
            f"compliance={info.get('mean_compliance',0)*100:.1f}%"
        )

    env.close()

    print("\n" + "═" * 55)
    print("  PHASE 4 DEMO COMPLETE")
    print("═" * 55)
    print(f"  Episodes         : {args.episodes}")
    print(f"  Mean reward      : {np.mean(episode_rewards):.3f}")
    print(f"  Reward range     : [{np.min(episode_rewards):.3f}, {np.max(episode_rewards):.3f}]")
    print(f"  ✅ Environment verified — zero errors")
    print("═" * 55)
    return 0


def cmd_train(args) -> int:
    """Full PPO training."""
    from phase4_rl.ppo_agent import PPOTrainer, PPOConfig

    cfg = PPOConfig(
        total_timesteps  = args.timesteps,
        learning_rate    = args.lr,
        n_steps          = args.n_steps,
        batch_size       = args.batch_size,
        sim_runs_per_step = args.sim_runs,
        horizon_hours    = args.horizon,
        use_curriculum   = not args.no_curriculum,
        checkpoint_dir   = args.checkpoint_dir,
        eval_episodes    = args.eval_episodes,
    )

    logger.info(f"Phase 4 Training — {cfg.total_timesteps:,} timesteps | "
                f"curriculum={'ON' if cfg.use_curriculum else 'OFF'}")

    trainer = PPOTrainer(cfg)
    model   = trainer.train()
    metrics = trainer.evaluate(model, n_episodes=cfg.eval_episodes)

    print("\n" + "═" * 55)
    print("  PHASE 4 TRAINING COMPLETE")
    print("═" * 55)
    print(f"  Timesteps        : {cfg.total_timesteps:,}")
    print(f"  Mean eval reward : {metrics['mean_episode_reward']:.3f} ± {metrics['std_episode_reward']:.3f}")
    print(f"  Mean breach rate : {metrics['mean_breach_rate']*100:.1f}%")
    print(f"  SLA compliance   : {metrics['mean_compliance_rate']*100:.1f}%")
    print(f"  Best model saved → {cfg.checkpoint_dir}/best_model.zip")
    print("═" * 55)
    return 0


def cmd_eval(args) -> int:
    """Evaluate a saved model."""
    from phase4_rl.ppo_agent import PPOTrainer, PPOConfig
    import numpy as np

    cfg = PPOConfig(
        sim_runs_per_step = args.sim_runs,
        eval_episodes     = args.episodes,
    )
    trainer = PPOTrainer(cfg)
    model   = PPOTrainer.load(args.model)
    metrics = trainer.evaluate(model, n_episodes=args.episodes)

    print("\n" + "═" * 55)
    print("  PHASE 4 EVALUATION RESULTS")
    print("═" * 55)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v}")
    print("═" * 55)
    return 0


def cmd_infer(args) -> int:
    """Generate a staffing recommendation."""
    from phase4_rl.rl_inference import RLInferenceEngine

    engine = RLInferenceEngine(
        model_path        = args.model,
        use_feature_store = args.use_feature_store,
    )

    if args.loop:
        engine.run_inference_loop(interval_seconds=args.interval)
    else:
        rec = engine.recommend()
        print("\n" + "═" * 65)
        print("  PHASE 4 RL RECOMMENDATION")
        print("═" * 65)
        print(f"  Timestamp   : {rec.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Confidence  : {rec.confidence*100:.1f}%")
        print(f"  Est. breach reduction : {rec.predicted_breach_reduction*100:.1f}%")
        print()
        print(f"  {rec.narrative}")
        print()
        print("  Agent changes:")
        for q in rec.actions:
            d = rec.actions[q]
            curr = rec.current_agents.get(q, 0)
            new  = rec.recommended_agents.get(q, curr)
            arrow = "→"
            if d > 0:
                sym = f"(+{d})"
            elif d < 0:
                sym = f"({d})"
            else:
                sym = "(=)"
            print(f"    {q:<25} {curr:>2} {arrow} {new:>2}  {sym}")
        print("═" * 65)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(argv=None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    dispatch = {
        "demo":  cmd_demo,
        "train": cmd_train,
        "eval":  cmd_eval,
        "infer": cmd_infer,
    }
    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
