"""Epsilon-Greedy exploration (Sutton & Barto, 2018).

The simplest exploration strategy: with probability epsilon, take a random action.
No intrinsic reward — exploration happens purely through action randomization.
PPO already has entropy-based exploration, so this provides zero intrinsic reward
and is included as a baseline / ablation.
"""

from __future__ import annotations

import torch

from rl_exploration_lab.exploration.base import BaseExploration


class EpsilonGreedy(BaseExploration):
    """Epsilon-Greedy exploration baseline.

    Returns zero intrinsic reward. Epsilon-greedy action selection is handled
    at the agent level (PPO's entropy bonus serves a similar purpose).

    This exists as a no-op exploration module for fair comparison:
    PPO + EpsilonGreedy = pure PPO baseline.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__(device=device)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Returns zero intrinsic reward (no exploration bonus)."""
        return torch.zeros(obs.shape[0], device=self.device)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """No-op update."""
        return {}
