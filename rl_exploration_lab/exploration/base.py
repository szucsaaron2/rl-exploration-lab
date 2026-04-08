"""Abstract base class for all exploration methods.

Every exploration method implements this interface so they're interchangeable
in the training loop. The PPO trainer calls compute_intrinsic_reward() during
rollout collection and update() after each PPO update.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseExploration(ABC):
    """Abstract base class for exploration methods.

    Subclasses must implement:
        - compute_intrinsic_reward: given observations, return intrinsic reward per sample
        - update: update any internal state (e.g., train predictor network)

    Optional overrides:
        - get_exploration_loss: return the exploration-specific loss for logging
        - state_dict / load_state_dict: for checkpointing
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    @abstractmethod
    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intrinsic reward for a batch of transitions.

        Args:
            obs: Current observations, shape (batch, obs_dim).
            next_obs: Next observations, shape (batch, obs_dim).
            action: Actions taken, shape (batch,).

        Returns:
            Intrinsic rewards, shape (batch,). Higher = more novel/interesting.
        """
        ...

    @abstractmethod
    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Update internal state (e.g., train predictor network).

        Called once per PPO update with the same batch used for the PPO update.

        Args:
            batch: Dict with keys 'obs', 'next_obs', 'action', etc.

        Returns:
            Dict of metrics for logging (e.g., {'exploration_loss': 0.05}).
        """
        ...

    def get_exploration_loss(self) -> float | None:
        """Return the current exploration loss for logging. Optional."""
        return None

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Load state from a checkpoint."""
        pass
