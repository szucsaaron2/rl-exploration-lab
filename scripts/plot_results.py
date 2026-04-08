#!/usr/bin/env python3
"""Generate comparison plots from saved benchmark results.

Usage:
    python scripts/plot_results.py --results results/ --output plots/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_exploration_lab.evaluation.metrics import AggregatedResult, format_results_table
from rl_exploration_lab.evaluation.plots import plot_method_comparison_bar


def load_results(results_dir: Path) -> list[AggregatedResult]:
    results = []
    for f in sorted(results_dir.glob("*.json")):
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


def main():
    parser = argparse.ArgumentParser(description="Generate plots from benchmark results")
    parser.add_argument(
        "--results", type=str, default="results",
        help="Results directory with JSON files.",
    )
    parser.add_argument("--output", type=str, default="plots", help="Output directory for plots.")
    args = parser.parse_args()

    results_dir = Path(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Loaded {len(results)} results\n")
    print(format_results_table(results))

    # Generate per-environment bar charts
    envs = sorted(set(r.env_name for r in results))
    for env_name in envs:
        env_results = [r for r in results if r.env_name == env_name]
        plot_method_comparison_bar(
            method_names=[r.method for r in env_results],
            mean_rewards=[r.mean_reward for r in env_results],
            std_rewards=[r.std_reward for r in env_results],
            env_name=env_name,
            output_path=output_dir / f"comparison_{env_name}.png",
        )

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
