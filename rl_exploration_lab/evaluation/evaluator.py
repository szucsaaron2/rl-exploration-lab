"""Evaluator for running experiments across multiple seeds.

Provides the main evaluation loop: for each seed, create env + policy + exploration,
train, collect results, and aggregate across seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rl_exploration_lab.envs.minigrid_wrapper import make_wrapped_env
from rl_exploration_lab.evaluation.metrics import (
    AggregatedResult,
    ExperimentResult,
    aggregate_results,
)
from rl_exploration_lab.exploration.amigo import AMIGo
from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.exploration.count_based import CountBased
from rl_exploration_lab.exploration.epsilon_greedy import EpsilonGreedy
from rl_exploration_lab.exploration.icm import ICM
from rl_exploration_lab.exploration.language.clip_noveld import CLIPNovelD
from rl_exploration_lab.exploration.language.clip_rnd import CLIPRND
from rl_exploration_lab.exploration.language.l_amigo import LAMIGo
from rl_exploration_lab.exploration.language.l_noveld import LNovelD
from rl_exploration_lab.exploration.language.semantic import SemanticExploration
from rl_exploration_lab.exploration.ngu import NGU
from rl_exploration_lab.exploration.noveld import NovelD
from rl_exploration_lab.exploration.ride import RIDE
from rl_exploration_lab.exploration.rnd import RND
from rl_exploration_lab.exploration.shelm.shelm_oracle import SHELMOracle
from rl_exploration_lab.exploration.shelm.shelm_rnd import SHELMRND
from rl_exploration_lab.exploration.ucb import UCB
from rl_exploration_lab.networks.policy import ActorCritic
from rl_exploration_lab.training.trainer import Trainer

# Registry of exploration methods
EXPLORATION_METHODS: dict[str, type] = {
    "none": EpsilonGreedy,
    "epsilon_greedy": EpsilonGreedy,
    "ucb": UCB,
    "count_based": CountBased,
    "rnd": RND,
    "icm": ICM,
    "ride": RIDE,
    "noveld": NovelD,
    "ngu": NGU,
    "amigo": AMIGo,
    "clip_rnd": CLIPRND,
    "clip_noveld": CLIPNovelD,
    "semantic": SemanticExploration,
    "l_noveld": LNovelD,
    "l_amigo": LAMIGo,
    "shelm_rnd": SHELMRND,
    "shelm_oracle": SHELMOracle,
}


def create_exploration(
    method_name: str, obs_dim: int = 147, device: str = "cpu", **kwargs
) -> BaseExploration:
    """Create an exploration method by name.

    Args:
        method_name: Name of the exploration method.
        obs_dim: Observation dimension.
        device: Torch device.
        **kwargs: Additional arguments passed to the exploration constructor.

    Returns:
        An exploration method instance.
    """
    if method_name not in EXPLORATION_METHODS:
        available = ", ".join(EXPLORATION_METHODS.keys())
        raise ValueError(f"Unknown exploration method '{method_name}'. Available: {available}")

    cls = EXPLORATION_METHODS[method_name]

    # Methods that accept obs_dim (neural network-based)
    obs_dim_methods = (
        RND, ICM, RIDE, NovelD, NGU, AMIGo,
        CLIPRND, CLIPNovelD, SemanticExploration, LNovelD, LAMIGo,
        SHELMRND, SHELMOracle,
    )
    # Methods that accept n_actions
    n_actions_methods = (ICM, RIDE)

    ctor_kwargs: dict = {"device": device, **kwargs}
    if cls in obs_dim_methods:
        ctor_kwargs["obs_dim"] = obs_dim
    if cls in n_actions_methods:
        ctor_kwargs.setdefault("n_actions", 7)  # MiniGrid default

    return cls(**ctor_kwargs)


def run_single_experiment(
    method_name: str,
    env_name: str,
    seed: int,
    config: dict,
    device: str = "cpu",
    log_dir: str | None = None,
) -> ExperimentResult:
    """Run a single training experiment (one method, one env, one seed).

    Args:
        method_name: Name of the exploration method.
        env_name: Environment short name or gymnasium ID.
        seed: Random seed.
        config: Training configuration dict.
        device: Torch device.
        log_dir: Directory for TensorBoard logs.

    Returns:
        ExperimentResult with all collected metrics.
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = make_wrapped_env(env_name, seed=seed)

    # Create policy
    policy = ActorCritic(
        obs_dim=env.obs_shape[0],
        n_actions=env.n_actions,
        hidden_dim=config.get("hidden_dim", 128),
        embed_dim=config.get("embed_dim", 64),
    )

    # Create exploration method
    exploration_kwargs = config.get("exploration_kwargs", {})
    exploration = create_exploration(
        method_name,
        obs_dim=env.obs_shape[0],
        device=device,
        **exploration_kwargs,
    )

    # Set up log directory
    if log_dir is None:
        log_dir = f"runs/{method_name}/{env_name}/seed_{seed}"

    # Create trainer and run
    trainer = Trainer(
        env=env,
        policy=policy,
        exploration=exploration,
        config=config,
        device=device,
        log_dir=log_dir,
    )

    stats = trainer.train()

    # Package results
    result = ExperimentResult(
        method=method_name,
        env_name=env_name,
        seed=seed,
        total_steps=stats.get("total_steps", 0),
        episode_rewards=trainer._episode_rewards,
        episode_lengths=trainer._episode_lengths,
        episode_solved=trainer._episode_solved,
        elapsed_seconds=stats.get("elapsed_seconds", 0.0),
    )

    env.close()
    return result


def run_evaluation(
    method_name: str,
    env_name: str,
    seeds: list[int],
    config: dict,
    device: str = "cpu",
    output_dir: str = "results",
) -> AggregatedResult:
    """Run evaluation across multiple seeds and aggregate.

    Args:
        method_name: Exploration method name.
        env_name: Environment name.
        seeds: List of random seeds to evaluate.
        config: Training configuration.
        device: Torch device.
        output_dir: Directory to save results.

    Returns:
        AggregatedResult across all seeds.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  {method_name} | {env_name} | seed={seed}")
        print(f"{'='*60}")

        log_dir = str(output_path / "tensorboard" / method_name / env_name / f"seed_{seed}")
        result = run_single_experiment(
            method_name=method_name,
            env_name=env_name,
            seed=seed,
            config=config,
            device=device,
            log_dir=log_dir,
        )
        results.append(result)

        print(f"  → Mean reward: {result.mean_reward:.4f} | "
              f"Solve rate: {result.solve_rate:.1%} | "
              f"Time: {result.elapsed_seconds:.1f}s")

    aggregated = aggregate_results(results)

    # Save results to JSON
    result_file = output_path / f"{method_name}_{env_name}.json"
    with open(result_file, "w") as f:
        json.dump({
            "method": aggregated.method,
            "env_name": aggregated.env_name,
            "n_seeds": aggregated.n_seeds,
            "mean_reward": aggregated.mean_reward,
            "std_reward": aggregated.std_reward,
            "stderr_reward": aggregated.stderr_reward,
            "mean_solve_rate": aggregated.mean_solve_rate,
            "mean_length": aggregated.mean_length,
            "per_seed_rewards": aggregated.per_seed_rewards,
        }, f, indent=2)

    print(f"\n  Aggregated: {aggregated.mean_reward:.4f} ± {aggregated.std_reward:.4f}")
    print(f"  Results saved to {result_file}")

    return aggregated
