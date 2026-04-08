"""Go-Explore Phase 2: Robustification via the Backward Algorithm.

Takes a brittle trajectory from Phase 1 and trains a robust policy via
imitation learning. Uses PPO with curriculum starting near the end of
the trajectory and gradually backing up to the start.

Based on Salimans & Chen (2018): "Learning Montezuma's Revenge from
a Single Demonstration."

Steps:
1. Start the agent near the last state in the trajectory
2. Train PPO to reach the same or higher reward from there
3. Once the agent succeeds reliably, back up the starting point
4. Repeat until the agent can perform from the initial state
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from rl_exploration_lab.envs.minigrid_wrapper import MiniGridWrapper
from rl_exploration_lab.networks.policy import ActorCritic
from rl_exploration_lab.training.ppo import PPOTrainer
from rl_exploration_lab.training.rollout import RolloutBuffer


class BackwardAlgorithm:
    """Robustify a brittle trajectory into a robust policy.

    Args:
        env: Wrapped MiniGrid environment.
        trajectory: Action sequence from Go-Explore Phase 1.
        policy: ActorCritic network to train.
        config: Training configuration dict.
        backup_interval: Steps of training before backing up the start point.
        success_threshold: Fraction of episodes that must succeed before backing up.
        backup_steps: How many trajectory steps to back up each time.
        device: Torch device.
    """

    def __init__(
        self,
        env: MiniGridWrapper,
        trajectory: list[int],
        policy: ActorCritic,
        config: dict,
        backup_interval: int = 50_000,
        success_threshold: float = 0.8,
        backup_steps: int = 5,
        device: str = "cpu",
    ):
        self.env = env
        self.trajectory = trajectory
        self.device = device
        self.backup_interval = backup_interval
        self.success_threshold = success_threshold
        self.backup_steps = backup_steps

        # Current starting point in the trajectory (starts near the end)
        self._start_point = max(0, len(trajectory) - backup_steps)

        # PPO trainer
        self.ppo = PPOTrainer(
            policy=policy,
            lr=config.get("lr", 2.5e-4),
            clip_range=config.get("clip_range", 0.2),
            value_coef=config.get("value_coef", 0.5),
            entropy_coef=config.get("entropy_coef", 0.01),
            n_epochs=config.get("n_epochs", 4),
            device=device,
        )
        self.policy = policy

        self.buffer = RolloutBuffer(
            buffer_size=config.get("rollout_steps", 2048),
            obs_dim=env.obs_shape[0],
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            intrinsic_coef=0.0,  # No intrinsic reward in Phase 2
            device=device,
        )

    def _replay_to_start_point(self) -> tuple[np.ndarray, bool]:
        """Replay the trajectory up to the current start point.

        Returns:
            (observation, success) — the obs at the start point, and whether
            replay succeeded without episode termination.
        """
        obs_np, _ = self.env.reset()
        for i, action in enumerate(self.trajectory[: self._start_point]):
            obs_np, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                return obs_np, False
        return obs_np, True

    def train(self, total_steps: int = 500_000, verbose: bool = True) -> dict:
        """Run the backward algorithm training loop.

        Args:
            total_steps: Total environment steps for robustification.
            verbose: Whether to show progress bar.

        Returns:
            Training statistics dict.
        """
        global_step = 0
        episodes_succeeded = 0
        episodes_total = 0
        recent_successes: list[bool] = []

        pbar = tqdm(total=total_steps, desc="Robustify (Phase 2)", disable=not verbose)

        while global_step < total_steps:
            # Replay to current start point
            obs_np, replay_ok = self._replay_to_start_point()
            if not replay_ok:
                # If replay fails, back up further
                self._start_point = max(0, self._start_point - self.backup_steps)
                continue

            obs = torch.from_numpy(obs_np).float().to(self.device)

            # Collect a rollout from the start point
            self.buffer.reset()
            episode_reward = 0.0

            for _ in range(self.buffer.buffer_size):
                with torch.no_grad():
                    action, log_prob, _, value = self.policy.get_action_and_value(
                        obs.unsqueeze(0)
                    )
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

                next_obs_np, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated
                next_obs = torch.from_numpy(next_obs_np).float().to(self.device)

                self.buffer.add(
                    obs=obs, next_obs=next_obs, action=action,
                    ext_reward=reward, int_reward=0.0, done=done,
                    log_prob=log_prob, value=value,
                )

                episode_reward += reward
                obs = next_obs
                global_step += 1
                pbar.update(1)

                if done:
                    episodes_total += 1
                    success = reward > 0
                    if success:
                        episodes_succeeded += 1
                    recent_successes.append(success)
                    if len(recent_successes) > 100:
                        recent_successes = recent_successes[-100:]

                    # Restart from start point
                    obs_np, replay_ok = self._replay_to_start_point()
                    if not replay_ok:
                        break
                    obs = torch.from_numpy(obs_np).float().to(self.device)

                if global_step >= total_steps:
                    break

            # PPO update
            with torch.no_grad():
                _, last_value = self.policy(obs.unsqueeze(0))
                last_value = last_value.squeeze()
            self.buffer.compute_advantages(last_value, done)
            batches = self.buffer.get_batches(64)
            self.ppo.update(batches)

            # Check if we should back up the starting point
            if recent_successes and len(recent_successes) >= 20:
                success_rate = sum(recent_successes[-20:]) / 20
                if success_rate >= self.success_threshold and self._start_point > 0:
                    self._start_point = max(0, self._start_point - self.backup_steps)
                    recent_successes.clear()
                    if verbose:
                        pbar.set_postfix(
                            start=self._start_point,
                            sr=f"{success_rate:.0%}",
                        )

        pbar.close()

        return {
            "total_steps": global_step,
            "episodes_total": episodes_total,
            "episodes_succeeded": episodes_succeeded,
            "final_start_point": self._start_point,
            "fully_robustified": self._start_point == 0,
            "final_success_rate": (
                sum(recent_successes) / len(recent_successes)
                if recent_successes else 0.0
            ),
        }
