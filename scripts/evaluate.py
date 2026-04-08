#!/usr/bin/env python3
"""Run the full benchmark: all methods × all environments × N seeds.

Usage:
    # Full benchmark (all methods, all envs, 5 seeds) — takes hours
    python scripts/evaluate.py --suite full

    # Quick benchmark (3 methods, 1 env, 2 seeds) — minutes
    python scripts/evaluate.py --suite quick

    # Custom: specific methods and envs
    python scripts/evaluate.py --methods rnd noveld clip_rnd --envs Empty-8x8 KeyCorridorS3R2 --seeds 0 1 2

    # Generate plots from existing results
    python scripts/evaluate.py --plot-only --results-dir results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_exploration_lab.evaluation.evaluator import (
    EXPLORATION_METHODS,
    run_evaluation,
)
from rl_exploration_lab.evaluation.metrics import AggregatedResult, format_results_table
from rl_exploration_lab.evaluation.plots import plot_method_comparison_bar

# Pre-defined benchmark suites
SUITES = {
    "quick": {
        "methods": ["none", "rnd", "count_based"],
        "envs": ["Empty-8x8"],
        "steps": 50_000,
        "seeds": [0, 1],
    },
    "medium": {
        "methods": ["none", "rnd", "icm", "noveld", "count_based", "ngu"],
        "envs": ["Empty-8x8", "DoorKey-6x6"],
        "steps": 200_000,
        "seeds": [0, 1, 2],
    },
    "full": {
        "methods": [
            "none", "ucb", "count_based", "rnd", "icm", "ride", "noveld",
            "ngu", "amigo", "clip_rnd", "clip_noveld", "semantic",
            "l_noveld", "l_amigo", "shelm_rnd", "shelm_oracle",
        ],
        "envs": ["Empty-8x8", "DoorKey-6x6", "KeyCorridorS3R2"],
        "steps": 2_000_000,
        "seeds": [0, 1, 2, 3, 4],
    },
}


def load_default_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_plots(all_results: list[AggregatedResult], output_dir: Path):
    """Generate comparison plots from aggregated results."""
    # Group by environment
    envs = sorted(set(r.env_name for r in all_results))

    for env_name in envs:
        env_results = [r for r in all_results if r.env_name == env_name]
        if not env_results:
            continue

        names = [r.method for r in env_results]
        means = [r.mean_reward for r in env_results]
        stds = [r.std_reward for r in env_results]

        plot_method_comparison_bar(
            method_names=names,
            mean_rewards=means,
            std_rewards=stds,
            env_name=env_name,
            output_path=output_dir / "plots" / f"comparison_{env_name}.png",
        )


def load_existing_results(results_dir: Path) -> list[AggregatedResult]:
    """Load previously saved JSON results."""
    results = []
    for f in results_dir.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
        results.append(AggregatedResult(
            method=data["method"],
            env_name=data["env_name"],
            n_seeds=data["n_seeds"],
            mean_reward=data["mean_reward"],
            std_reward=data["std_reward"],
            stderr_reward=data.get("stderr_reward", 0.0),
            mean_solve_rate=data.get("mean_solve_rate", 0.0),
            mean_length=data.get("mean_length", 0.0),
            mean_exploration_loss=data.get("mean_exploration_loss"),
            per_seed_rewards=data.get("per_seed_rewards", []),
        ))
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="RL Exploration Lab — Full Benchmark")
    parser.add_argument("--suite", type=str, choices=list(SUITES.keys()), default=None,
                        help="Pre-defined benchmark suite.")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="Specific methods to evaluate.")
    parser.add_argument("--envs", type=str, nargs="+", default=None,
                        help="Specific environments to evaluate.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="results")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only generate plots from existing results.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory with existing JSON results (for --plot-only).")
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot-only mode
    if args.plot_only:
        rdir = Path(args.results_dir) if args.results_dir else output_dir
        all_results = load_existing_results(rdir)
        if not all_results:
            print(f"No results found in {rdir}")
            return
        print(f"\nLoaded {len(all_results)} results from {rdir}")
        print("\n" + format_results_table(all_results))
        generate_plots(all_results, output_dir)
        return

    # Determine what to run
    if args.suite:
        suite = SUITES[args.suite]
        methods = args.methods or suite["methods"]
        envs = args.envs or suite["envs"]
        seeds = args.seeds or suite["seeds"]
        steps = args.steps or suite["steps"]
    else:
        methods = args.methods or ["none", "rnd"]
        envs = args.envs or ["Empty-8x8"]
        seeds = args.seeds or [0, 1, 2]
        steps = args.steps or 200_000

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    config = load_default_config()
    config["total_steps"] = steps

    total_runs = len(methods) * len(envs) * len(seeds)
    print(f"\n{'='*60}")
    print(f"  RL Exploration Lab — Full Benchmark")
    print(f"  Methods:      {len(methods)} ({', '.join(methods[:5])}{'...' if len(methods) > 5 else ''})")
    print(f"  Environments: {len(envs)} ({', '.join(envs)})")
    print(f"  Seeds:        {len(seeds)}")
    print(f"  Steps/run:    {steps:,}")
    print(f"  Total runs:   {total_runs}")
    print(f"  Device:       {device}")
    print(f"{'='*60}\n")

    all_results: list[AggregatedResult] = []

    for env_name in envs:
        for method_name in methods:
            config_copy = dict(config)
            config_copy["env_name"] = env_name
            config_copy["exploration_method"] = method_name

            try:
                result = run_evaluation(
                    method_name=method_name,
                    env_name=env_name,
                    seeds=seeds,
                    config=config_copy,
                    device=device,
                    output_dir=str(output_dir),
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n  ❌ FAILED: {method_name} on {env_name}: {e}\n")

    # Print final summary
    if all_results:
        print(f"\n{'='*60}")
        print(f"  FINAL RESULTS")
        print(f"{'='*60}\n")
        print(format_results_table(all_results))

        # Save summary
        summary_path = output_dir / "summary.md"
        with open(summary_path, "w") as f:
            f.write(f"# Benchmark Results\n\n")
            f.write(f"Steps per run: {steps:,}\n\n")
            f.write(format_results_table(all_results))
            f.write("\n")
        print(f"\nSummary saved to {summary_path}")

        # Generate plots
        generate_plots(all_results, output_dir)


if __name__ == "__main__":
    main()
