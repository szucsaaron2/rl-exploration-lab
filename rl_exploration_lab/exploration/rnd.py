"""Random Network Distillation (Burda et al., 2018).

Two networks: a fixed random target network and a trainable predictor.
The intrinsic reward is the prediction error between them.
Novel states produce high prediction error → high intrinsic reward.

This is the core exploration method from the thesis and a strong baseline.
r_i = ||f_target(s) - f_predictor(s)||^2
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.predictors import RNDModule


class RND(BaseExploration):
    """Random Network Distillation exploration.

    Args:
        obs_dim: Observation dimension.
        output_dim: RND embedding dimension.
        hidden_dim: Hidden layer size for target/predictor networks.
        n_layers: Number of hidden layers per network.
        lr: Learning rate for the predictor network.
        reward_clip: Clip intrinsic reward to [0, reward_clip]. None for no clipping.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        output_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        lr: float = 1e-3,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        self.rnd = RNDModule(obs_dim, output_dim, hidden_dim, n_layers).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # Running statistics for reward normalization
        self._reward_running_mean = 0.0
        self._reward_running_var = 1.0
        self._reward_count = 0

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RND intrinsic reward based on next-state prediction error.

        We use next_obs (the state we arrived at) as the input to RND,
        following the standard formulation.
        """
        with torch.no_grad():
            reward = self.rnd.compute_intrinsic_reward(next_obs)

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)

        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the predictor network to match the target on observed states."""
        next_obs = batch["next_obs"]

        loss = self.rnd.compute_loss(next_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()
        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {
            "rnd": self.rnd.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.rnd.load_state_dict(state["rnd"])
        self.optimizer.load_state_dict(state["optimizer"])
