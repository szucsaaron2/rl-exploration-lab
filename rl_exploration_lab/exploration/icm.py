"""Intrinsic Curiosity Module (Pathak et al., 2017).

Uses a forward and inverse dynamics model. The intrinsic reward is the
prediction error of the forward model:
    r_i = η/2 * ||φ(s_{t+1}) - φ'(s_{t+1})||^2

where φ(s) is the true feature encoding and φ'(s) is the predicted encoding.

The inverse model ensures the learned features capture action-relevant
information (not random noise), preventing the "noisy TV" problem somewhat.

Total loss = (1 - β) * L_inverse + β * L_forward
where β controls the relative weight of forward vs inverse model learning.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.dynamics import DynamicsModel


class ICM(BaseExploration):
    """Intrinsic Curiosity Module exploration.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        embed_dim: Feature embedding dimension.
        hidden_dim: Hidden layer size.
        lr: Learning rate for the dynamics model.
        eta: Scaling factor for intrinsic reward.
        beta: Weight for forward vs inverse loss (beta=0.2 → 80% inverse, 20% forward).
        reward_clip: Maximum intrinsic reward. None for no clipping.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        n_actions: int = 7,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        eta: float = 0.01,
        beta: float = 0.2,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.eta = eta
        self.beta = beta
        self.reward_clip = reward_clip
        self._last_forward_loss = 0.0
        self._last_inverse_loss = 0.0

        self.dynamics = DynamicsModel(obs_dim, n_actions, embed_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute curiosity-driven intrinsic reward (forward prediction error)."""
        with torch.no_grad():
            _, _, intrinsic_reward = self.dynamics.compute_icm_losses(obs, next_obs, action)
            intrinsic_reward = self.eta * intrinsic_reward

        if self.reward_clip is not None:
            intrinsic_reward = intrinsic_reward.clamp(0.0, self.reward_clip)

        return intrinsic_reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the dynamics model on the rollout batch."""
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        forward_loss, inverse_loss, _ = self.dynamics.compute_icm_losses(obs, next_obs, actions)
        total_loss = self.beta * forward_loss + (1 - self.beta) * inverse_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._last_forward_loss = forward_loss.item()
        self._last_inverse_loss = inverse_loss.item()

        return {
            "forward_loss": self._last_forward_loss,
            "inverse_loss": self._last_inverse_loss,
            "exploration_loss": total_loss.item(),
        }

    def get_exploration_loss(self) -> float | None:
        return self._last_forward_loss

    def state_dict(self) -> dict:
        return {
            "dynamics": self.dynamics.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dynamics.load_state_dict(state["dynamics"])
        self.optimizer.load_state_dict(state["optimizer"])
