"""Plotting utilities for thesis-style comparison figures.

Generates:
- Reward curves with standard error shading (like thesis Figure 5.1)
- Exploration loss curves (like thesis Figure 5.2)
- Bar chart comparisons across methods
- Results summary tables
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# Consistent colors for each method family
METHOD_COLORS = {
    "none": "#808080",
    "epsilon_greedy": "#808080",
    "ucb": "#A0A0A0",
    "count_based": "#D2691E",
    "rnd": "#2196F3",
    "icm": "#4CAF50",
    "ride": "#FF9800",
    "noveld": "#9C27B0",
    "ngu": "#E91E63",
    "amigo": "#00BCD4",
    "clip_rnd": "#1565C0",
    "clip_noveld": "#6A1B9A",
    "semantic": "#2E7D32",
    "l_noveld": "#7B1FA2",
    "l_amigo": "#00838F",
    "shelm_rnd": "#C62828",
    "shelm_oracle": "#AD1457",
    "go_explore": "#FF6F00",
}


def plot_reward_curves(
    results: dict[str, list[list[float]]],
    title: str = "Mean Episodic Reward Across Multiple Runs",
    output_path: str | Path = "plots/reward_curves.png",
    window: int = 100,
    figsize: tuple[float, float] = (12, 7),
):
    """Plot reward curves with standard error shading (thesis Figure 5.1 style).

    Args:
        results: Dict mapping method_name → list of per-seed episode reward lists.
                 e.g. {"rnd": [[0.1, 0.2, ...], [0.05, 0.15, ...], ...]}
        title: Plot title.
        output_path: Where to save the figure.
        window: Smoothing window size for the rolling mean.
        figsize: Figure dimensions.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for method_name, seed_rewards in results.items():
        if not seed_rewards or not seed_rewards[0]:
            continue

        # Smooth each seed's rewards with a rolling mean
        smoothed = []
        min_len = min(len(s) for s in seed_rewards)
        for seed_data in seed_rewards:
            data = np.array(seed_data[:min_len])
            if len(data) >= window:
                kernel = np.ones(window) / window
                smooth = np.convolve(data, kernel, mode="valid")
            else:
                smooth = data
            smoothed.append(smooth)

        smoothed = np.array(smoothed)
        mean = smoothed.mean(axis=0)
        stderr = smoothed.std(axis=0) / np.sqrt(len(smoothed))
        steps = np.arange(len(mean))

        color = METHOD_COLORS.get(method_name, None)
        ax.plot(steps, mean, label=method_name, color=color, linewidth=1.5)
        ax.fill_between(steps, mean - stderr, mean + stderr, alpha=0.2, color=color)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Mean Episodic Reward", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward curves to {output_path}")


def plot_exploration_loss(
    results: dict[str, list[list[float]]],
    title: str = "Exploration Loss Across Multiple Runs",
    output_path: str | Path = "plots/exploration_loss.png",
    figsize: tuple[float, float] = (12, 7),
):
    """Plot exploration loss curves (thesis Figure 5.2 style).

    Args:
        results: Dict mapping method_name → list of per-seed loss lists.
        title: Plot title.
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    for method_name, seed_losses in results.items():
        if not seed_losses or not seed_losses[0]:
            continue

        min_len = min(len(s) for s in seed_losses)
        data = np.array([s[:min_len] for s in seed_losses])
        mean = data.mean(axis=0)
        stderr = data.std(axis=0) / np.sqrt(len(data))
        steps = np.arange(len(mean))

        color = METHOD_COLORS.get(method_name, None)
        ax.plot(steps, mean, label=method_name, color=color, linewidth=1.5)
        ax.fill_between(steps, mean - stderr, mean + stderr, alpha=0.2, color=color)

    ax.set_xlabel("Update Step", fontsize=12)
    ax.set_ylabel("Exploration Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved exploration loss curves to {output_path}")


def plot_method_comparison_bar(
    method_names: list[str],
    mean_rewards: list[float],
    std_rewards: list[float],
    env_name: str = "KeyCorridorS3R2",
    output_path: str | Path = "plots/comparison_bar.png",
    figsize: tuple[float, float] = (14, 6),
):
    """Bar chart comparing final mean rewards across methods.

    Args:
        method_names: List of method names.
        mean_rewards: Mean reward for each method.
        std_rewards: Standard deviation for each method.
        env_name: Environment name for the title.
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Sort by mean reward
    sorted_idx = np.argsort(mean_rewards)[::-1]
    names = [method_names[i] for i in sorted_idx]
    means = [mean_rewards[i] for i in sorted_idx]
    stds = [std_rewards[i] for i in sorted_idx]
    colors = [METHOD_COLORS.get(n, "#888888") for n in names]

    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Episodic Reward", fontsize=12)
    ax.set_title(f"Method Comparison — {env_name}", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison bar chart to {output_path}")
