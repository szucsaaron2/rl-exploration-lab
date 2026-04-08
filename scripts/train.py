#!/usr/bin/env python3
"""Main training script for rl-exploration-lab.

Usage:
    # Quick run with defaults (PPO baseline on KeyCorridorS3R2, 1 seed)
    python scripts/train.py

    # Run RND with 5 seeds
    python scripts/train.py --method rnd --seeds 0 1 2 3 4

    # Run from config file
    python scripts/train.py --config configs/experiments/rnd_keycorridor.yaml

    # Short test run
    python scripts/train.py --method rnd --env Empty-8x8 --steps 50000 --seeds 0

    # Run Go-Explore (Phase 1 — no GPU needed)
    python scripts/train.py --method go_explore --env KeyCorridorS3R2 --steps 100000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_exploration_lab.evaluation.evaluator import run_evaluation, run_single_experiment


def load_config(config_path: str | None) -> dict:
    """Load config from YAML file, falling back to defaults."""
    default_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(default_path) as f:
        config = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            overrides = yaml.safe_load(f)
        if overrides:
            config.update(overrides)

    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL Exploration Lab — Training Script")

    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides defaults).",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Exploration method: none, epsilon_greedy, count_based, rnd.",
    )
    parser.add_argument(
        "--env", type=str, default=None,
        help="Environment name (e.g. KeyCorridorS3R2, Empty-8x8, DoorKey-6x6).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Total environment steps.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Random seeds (e.g. --seeds 0 1 2 3 4).",
    )
    parser.add_argument(
        "--intrinsic-coef", type=float, default=None,
        help="Intrinsic reward coefficient beta.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device (cpu, cuda, cuda:0, etc.).",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for results.",
    )

    return parser.parse_args()


def _run_go_explore(config: dict, seeds: list[int], device: str):
    """Run Go-Explore Phase 1 (archive-based exploration, no neural network).

    Go-Explore has its own training loop that doesn't use PPO.
    It returns the best trajectory found and archive statistics.
    """
    from rl_exploration_lab.envs.minigrid_wrapper import make_wrapped_env
    from rl_exploration_lab.exploration.go_explore.cell_repr import MiniGridDomainCell
    from rl_exploration_lab.exploration.go_explore.go_explore import GoExplorePhase1

    env_name = config["env_name"]
    total_steps = config["total_steps"]

    for seed in seeds:
        np.random.seed(seed)
        env = make_wrapped_env(env_name, seed=seed)

        explorer = GoExplorePhase1(
            env=env,
            cell_repr=MiniGridDomainCell(),
            explore_steps=config.get("go_explore_steps", 100),
            action_repeat_prob=config.get("go_explore_sticky", 0.95),
            total_steps=total_steps,
        )

        results = explorer.run(verbose=True)

        print(f"\n{'='*60}")
        print(f"  GO-EXPLORE RESULTS (seed={seed})")
        print(f"  Archive size:    {results['archive_size']}")
        print(f"  Best score:      {results['best_score']:.4f}")
        print(f"  Best traj len:   {results['best_trajectory_length']}")
        print(f"  Episodes:        {results['episodes_completed']}")
        print(f"  Total steps:     {results['total_steps']:,}")
        print(f"{'='*60}")

        print(f"\nmean_reward: {results['best_score']:.6f}")
        print(f"archive_size: {results['archive_size']}")

        env.close()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.method is not None:
        config["exploration_method"] = args.method
    if args.env is not None:
        config["env_name"] = args.env
    if args.steps is not None:
        config["total_steps"] = args.steps
    if args.seeds is not None:
        config["seeds"] = args.seeds
    if args.intrinsic_coef is not None:
        config["intrinsic_coef"] = args.intrinsic_coef

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    method = config["exploration_method"]
    env_name = config["env_name"]
    seeds = config.get("seeds", [0])

    print(f"\n{'='*60}")
    print("  RL Exploration Lab")
    print(f"  Method:      {method}")
    print(f"  Environment: {env_name}")
    print(f"  Seeds:       {seeds}")
    print(f"  Steps:       {config['total_steps']:,}")
    print(f"  Device:      {device}")
    if method != "go_explore":
        print(f"  Intrinsic β: {config.get('intrinsic_coef', 0.01)}")
    print(f"{'='*60}\n")

    # === Go-Explore has its own training loop (no PPO in Phase 1) ===
    if method == "go_explore":
        _run_go_explore(config, seeds, device)
        return

    if len(seeds) == 1:
        # Single seed — quick run
        result = run_single_experiment(
            method_name=method,
            env_name=env_name,
            seed=seeds[0],
            config=config,
            device=device,
            log_dir=f"{args.output}/tensorboard/{method}/{env_name}/seed_{seeds[0]}",
        )
        print(f"\n{'='*60}")
        print("  RESULTS")
        print(f"  Mean reward:  {result.mean_reward:.4f}")
        print(f"  Solve rate:   {result.solve_rate:.1%}")
        print(f"  Episodes:     {len(result.episode_rewards)}")
        print(f"  Time:         {result.elapsed_seconds:.1f}s")
        print(f"{'='*60}")

        # Print final metrics for grep-friendly output (for future autoresearch)
        print(f"\nmean_reward: {result.mean_reward:.6f}")
        print(f"std_reward: {result.std_reward:.6f}")
        print(f"solve_rate: {result.solve_rate:.4f}")
        if result.final_exploration_loss is not None:
            print(f"exploration_loss: {result.final_exploration_loss:.6f}")

    else:
        # Multi-seed evaluation
        aggregated = run_evaluation(
            method_name=method,
            env_name=env_name,
            seeds=seeds,
            config=config,
            device=device,
            output_dir=args.output,
        )

        print(f"\n{'='*60}")
        print(f"  AGGREGATED RESULTS ({aggregated.n_seeds} seeds)")
        print(f"  Mean reward:  {aggregated.mean_reward:.4f} ± {aggregated.std_reward:.4f}")
        print(f"  Stderr:       {aggregated.stderr_reward:.4f}")
        print(f"  Solve rate:   {aggregated.mean_solve_rate:.1%}")
        print(f"{'='*60}")

        print(f"\nmean_reward: {aggregated.mean_reward:.6f}")
        print(f"std_reward: {aggregated.std_reward:.6f}")


if __name__ == "__main__":
    main()
