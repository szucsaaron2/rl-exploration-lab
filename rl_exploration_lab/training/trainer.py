"""High-level training orchestrator.

Manages the full training loop: environment interaction → rollout collection →
intrinsic reward computation → PPO update → exploration method update → logging.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_exploration_lab.envs.minigrid_wrapper import MiniGridWrapper
from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.policy import ActorCritic
from rl_exploration_lab.training.ppo import PPOTrainer
from rl_exploration_lab.training.rollout import RolloutBuffer


class Trainer:
    """Orchestrates the full training loop.

    Args:
        env: Wrapped MiniGrid environment.
        policy: ActorCritic network.
        exploration: Exploration method (provides intrinsic rewards).
        config: Training configuration dict.
        device: Torch device string.
        log_dir: Directory for TensorBoard logs and checkpoints.
    """

    def __init__(
        self,
        env: MiniGridWrapper,
        policy: ActorCritic,
        exploration: BaseExploration,
        config: dict,
        device: str = "cpu",
        log_dir: str = "runs",
    ):
        self.env = env
        self.policy = policy
        self.exploration = exploration
        self.config = config
        self.device = device

        # Training hyperparameters
        self.total_steps = config.get("total_steps", 2_000_000)
        self.rollout_steps = config.get("rollout_steps", 2048)
        self.batch_size = config.get("batch_size", 64)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.intrinsic_coef = config.get("intrinsic_coef", 0.01)

        # PPO trainer
        self.ppo = PPOTrainer(
            policy=policy,
            lr=config.get("lr", 2.5e-4),
            clip_range=config.get("clip_range", 0.2),
            value_coef=config.get("value_coef", 0.5),
            entropy_coef=config.get("entropy_coef", 0.01),
            max_grad_norm=config.get("max_grad_norm", 0.5),
            n_epochs=config.get("n_epochs", 4),
            device=device,
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            obs_dim=env.obs_shape[0],
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            intrinsic_coef=self.intrinsic_coef,
            device=device,
        )

        # Logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Episode tracking
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._episode_solved: list[bool] = []

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Final training statistics dict.
        """
        obs_np, _ = self.env.reset()
        obs = torch.from_numpy(obs_np).float().to(self.device)
        global_step = 0
        n_updates = 0
        start_time = time.time()

        pbar = tqdm(total=self.total_steps, desc="Training", unit="steps")

        while global_step < self.total_steps:
            # === Collect rollout ===
            self.buffer.reset()
            for _ in range(self.rollout_steps):
                with torch.no_grad():
                    action, log_prob, _, value = self.policy.get_action_and_value(
                        obs.unsqueeze(0)
                    )
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
                value = value.squeeze(0)

                # Step environment
                next_obs_np, ext_reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                done = terminated or truncated
                next_obs = torch.from_numpy(next_obs_np).float().to(self.device)

                # Compute intrinsic reward
                with torch.no_grad():
                    int_reward = self.exploration.compute_intrinsic_reward(
                        obs.unsqueeze(0), next_obs.unsqueeze(0), action.unsqueeze(0)
                    ).squeeze(0)

                # Store transition
                self.buffer.add(
                    obs=obs,
                    next_obs=next_obs,
                    action=action,
                    ext_reward=ext_reward,
                    int_reward=int_reward.item(),
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )

                obs = next_obs
                global_step += 1
                pbar.update(1)

                # Track episode statistics
                if done:
                    if "episode" in info:
                        ep = info["episode"]
                        self._episode_rewards.append(ep["r"])
                        self._episode_lengths.append(ep["l"])
                        self._episode_solved.append(ep["solved"])
                    obs_np, _ = self.env.reset()
                    obs = torch.from_numpy(obs_np).float().to(self.device)

                if global_step >= self.total_steps:
                    break

            # === Compute advantages ===
            with torch.no_grad():
                _, last_value = self.policy(obs.unsqueeze(0))
                last_value = last_value.squeeze()
            self.buffer.compute_advantages(last_value, done)

            # === PPO update ===
            batches = self.buffer.get_batches(self.batch_size)
            ppo_metrics = self.ppo.update(batches)

            # === Exploration method update ===
            exploration_metrics = self.exploration.update({
                "obs": self.buffer.obs,
                "next_obs": self.buffer.next_obs,
                "actions": self.buffer.actions,
            })

            n_updates += 1

            # === Logging ===
            self._log_metrics(global_step, ppo_metrics, exploration_metrics)

        pbar.close()
        elapsed = time.time() - start_time

        final_stats = self._compute_final_stats(global_step, elapsed, n_updates)
        self.writer.close()
        return final_stats

    def _log_metrics(
        self,
        global_step: int,
        ppo_metrics: dict[str, float],
        exploration_metrics: dict[str, float],
    ):
        """Log metrics to TensorBoard and print summary."""
        # PPO metrics
        for key, val in ppo_metrics.items():
            self.writer.add_scalar(f"ppo/{key}", val, global_step)

        # Exploration metrics
        for key, val in exploration_metrics.items():
            self.writer.add_scalar(f"exploration/{key}", val, global_step)

        # Episode metrics (if we have recent episodes)
        if self._episode_rewards:
            recent_n = min(100, len(self._episode_rewards))
            recent_rewards = self._episode_rewards[-recent_n:]
            recent_lengths = self._episode_lengths[-recent_n:]
            recent_solved = self._episode_solved[-recent_n:]

            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)
            solve_rate = np.mean(recent_solved)

            self.writer.add_scalar("episode/mean_reward", mean_reward, global_step)
            self.writer.add_scalar("episode/mean_length", mean_length, global_step)
            self.writer.add_scalar("episode/solve_rate", solve_rate, global_step)

    def _compute_final_stats(
        self, global_step: int, elapsed: float, n_updates: int
    ) -> dict:
        """Compute final training statistics."""
        stats = {
            "total_steps": global_step,
            "total_updates": n_updates,
            "total_episodes": len(self._episode_rewards),
            "elapsed_seconds": elapsed,
            "steps_per_second": global_step / elapsed if elapsed > 0 else 0,
        }

        if self._episode_rewards:
            stats["mean_reward"] = float(np.mean(self._episode_rewards))
            stats["std_reward"] = float(np.std(self._episode_rewards))
            stats["mean_length"] = float(np.mean(self._episode_lengths))
            stats["solve_rate"] = float(np.mean(self._episode_solved))

            # Last 100 episodes
            recent = self._episode_rewards[-100:]
            stats["mean_reward_last100"] = float(np.mean(recent))
            stats["std_reward_last100"] = float(np.std(recent))

        exploration_loss = self.exploration.get_exploration_loss()
        if exploration_loss is not None:
            stats["exploration_loss"] = exploration_loss

        return stats

    def save_checkpoint(self, path: str):
        """Save a training checkpoint."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "ppo_optimizer": self.ppo.optimizer.state_dict(),
            "exploration": self.exploration.state_dict(),
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.ppo.optimizer.load_state_dict(checkpoint["ppo_optimizer"])
        self.exploration.load_state_dict(checkpoint["exploration"])
