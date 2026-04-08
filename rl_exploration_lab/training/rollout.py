"""Rollout buffer for PPO with intrinsic reward support.

Stores transitions collected during environment interaction and computes
advantages using Generalized Advantage Estimation (GAE).
"""

from __future__ import annotations

import torch
import numpy as np


class RolloutBuffer:
    """Stores rollout data and computes GAE advantages.

    Args:
        buffer_size: Number of steps to store before a PPO update.
        obs_dim: Observation dimension.
        gamma: Discount factor for extrinsic rewards.
        gae_lambda: GAE lambda parameter.
        intrinsic_gamma: Discount factor for intrinsic rewards (often same as gamma).
        intrinsic_coef: Coefficient beta for intrinsic reward: r_total = r_ext + beta * r_int.
        device: Torch device.
    """

    def __init__(
        self,
        buffer_size: int = 2048,
        obs_dim: int = 147,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        intrinsic_gamma: float = 0.99,
        intrinsic_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.intrinsic_gamma = intrinsic_gamma
        self.intrinsic_coef = intrinsic_coef
        self.device = device

        self._ptr = 0
        self._full = False
        self._allocate()

    def _allocate(self):
        """Pre-allocate tensors for the buffer."""
        self.obs = torch.zeros(self.buffer_size, self.obs_dim, device=self.device)
        self.next_obs = torch.zeros(self.buffer_size, self.obs_dim, device=self.device)
        self.actions = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)
        self.ext_rewards = torch.zeros(self.buffer_size, device=self.device)
        self.int_rewards = torch.zeros(self.buffer_size, device=self.device)
        self.dones = torch.zeros(self.buffer_size, device=self.device)
        self.log_probs = torch.zeros(self.buffer_size, device=self.device)
        self.values = torch.zeros(self.buffer_size, device=self.device)
        self.advantages = torch.zeros(self.buffer_size, device=self.device)
        self.returns = torch.zeros(self.buffer_size, device=self.device)

    def add(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        ext_reward: float,
        int_reward: float,
        done: bool,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ):
        """Add a single transition to the buffer."""
        idx = self._ptr
        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = action
        self.ext_rewards[idx] = ext_reward
        self.int_rewards[idx] = int_reward
        self.dones[idx] = float(done)
        self.log_probs[idx] = log_prob.detach()
        self.values[idx] = value.detach()

        self._ptr += 1
        if self._ptr >= self.buffer_size:
            self._full = True

    @property
    def is_full(self) -> bool:
        return self._full

    @property
    def size(self) -> int:
        return self.buffer_size if self._full else self._ptr

    def compute_advantages(self, last_value: torch.Tensor, last_done: bool):
        """Compute GAE advantages and returns using combined rewards.

        Args:
            last_value: Value estimate for the state after the last stored transition.
            last_done: Whether the last state was terminal.
        """
        # Combined reward: r_total = r_ext + beta * r_int
        combined_rewards = self.ext_rewards + self.intrinsic_coef * self.int_rewards

        last_gae = 0.0
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value.detach()
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                combined_rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int) -> list[dict[str, torch.Tensor]]:
        """Split the buffer into mini-batches for PPO updates.

        Args:
            batch_size: Size of each mini-batch.

        Returns:
            List of dicts, each containing a mini-batch of transitions.
        """
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)

        batches = []
        for start in range(0, self.buffer_size, batch_size):
            end = min(start + batch_size, self.buffer_size)
            idx = indices[start:end]
            idx_tensor = torch.tensor(idx, dtype=torch.long, device=self.device)

            batches.append({
                "obs": self.obs[idx_tensor],
                "next_obs": self.next_obs[idx_tensor],
                "actions": self.actions[idx_tensor],
                "log_probs": self.log_probs[idx_tensor],
                "values": self.values[idx_tensor],
                "advantages": self.advantages[idx_tensor],
                "returns": self.returns[idx_tensor],
            })

        return batches

    def reset(self):
        """Reset the buffer pointer for the next rollout."""
        self._ptr = 0
        self._full = False
