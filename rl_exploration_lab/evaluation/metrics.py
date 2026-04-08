"""Evaluation metrics for comparing exploration methods.

Provides functions to compute and aggregate metrics across seeds and environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ExperimentResult:
    """Results from a single experiment (one method, one environment, one seed)."""

    method: str
    env_name: str
    seed: int
    total_steps: int = 0
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    episode_solved: list[bool] = field(default_factory=list)
    exploration_losses: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.episode_rewards)) if self.episode_rewards else 0.0

    @property
    def solve_rate(self) -> float:
        return float(np.mean(self.episode_solved)) if self.episode_solved else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0

    @property
    def final_exploration_loss(self) -> float | None:
        return self.exploration_losses[-1] if self.exploration_losses else None


@dataclass
class AggregatedResult:
    """Results aggregated across multiple seeds for one method + environment."""

    method: str
    env_name: str
    n_seeds: int
    mean_reward: float
    std_reward: float
    stderr_reward: float
    mean_solve_rate: float
    mean_length: float
    mean_exploration_loss: float | None
    per_seed_rewards: list[float] = field(default_factory=list)


def aggregate_results(results: list[ExperimentResult]) -> AggregatedResult:
    """Aggregate results across seeds for one method + environment.

    Args:
        results: List of ExperimentResult from different seeds (same method + env).

    Returns:
        AggregatedResult with mean, std, and stderr across seeds.
    """
    assert len(results) > 0, "No results to aggregate"
    method = results[0].method
    env_name = results[0].env_name

    per_seed_rewards = [r.mean_reward for r in results]
    per_seed_solve_rates = [r.solve_rate for r in results]
    per_seed_lengths = [r.mean_length for r in results]
    per_seed_losses = [
        r.final_exploration_loss
        for r in results
        if r.final_exploration_loss is not None
    ]

    n_seeds = len(results)
    mean_reward = float(np.mean(per_seed_rewards))
    std_reward = float(np.std(per_seed_rewards))
    stderr_reward = std_reward / (n_seeds ** 0.5) if n_seeds > 1 else 0.0

    return AggregatedResult(
        method=method,
        env_name=env_name,
        n_seeds=n_seeds,
        mean_reward=mean_reward,
        std_reward=std_reward,
        stderr_reward=stderr_reward,
        mean_solve_rate=float(np.mean(per_seed_solve_rates)),
        mean_length=float(np.mean(per_seed_lengths)),
        mean_exploration_loss=(
            float(np.mean(per_seed_losses)) if per_seed_losses else None
        ),
        per_seed_rewards=per_seed_rewards,
    )


def format_results_table(results: list[AggregatedResult]) -> str:
    """Format aggregated results as a markdown table.

    Args:
        results: List of AggregatedResult objects.

    Returns:
        Markdown-formatted comparison table string.
    """
    # Sort by mean reward (descending)
    results = sorted(results, key=lambda r: r.mean_reward, reverse=True)

    lines = [
        "| Rank | Method | Environment | Mean Reward | ± Std | Solve Rate | Expl. Loss |",
        "|------|--------|-------------|-------------|-------|------------|------------|",
    ]

    for i, r in enumerate(results, 1):
        loss_str = f"{r.mean_exploration_loss:.4f}" if r.mean_exploration_loss is not None else "N/A"
        lines.append(
            f"| {i} | {r.method} | {r.env_name} | {r.mean_reward:.4f} | "
            f"± {r.std_reward:.4f} | {r.mean_solve_rate:.1%} | {loss_str} |"
        )

    return "\n".join(lines)
