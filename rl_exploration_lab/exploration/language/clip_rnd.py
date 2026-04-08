"""CLIP-RND: Random Network Distillation on CLIP-encoded observations.

Instead of running RND on raw observations, first encode them through CLIP's
vision encoder, then apply RND on the CLIP embeddings. This provides
language-grounded state representations for exploration.

From the thesis (§5.1): CLIP RND was one of the best-performing baselines,
reinforcing that language abstraction can be useful for efficient exploration.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.encoders import CLIPEncoder
from rl_exploration_lab.networks.predictors import RNDModule


class CLIPRND(BaseExploration):
    """RND exploration using CLIP-encoded observations.

    Args:
        obs_dim: Raw observation dimension (147 for MiniGrid).
        clip_model: CLIP model name.
        rnd_output_dim: RND embedding output dimension.
        rnd_hidden_dim: RND hidden layer size.
        rnd_n_layers: Number of RND hidden layers.
        lr: Predictor learning rate.
        reward_clip: Max intrinsic reward.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        clip_model: str = "ViT-B/32",
        rnd_output_dim: int = 64,
        rnd_hidden_dim: int = 128,
        rnd_n_layers: int = 2,
        lr: float = 1e-3,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        # CLIP encoder (frozen)
        self.clip = CLIPEncoder(model_name=clip_model, device=device)

        # RND operates on CLIP embeddings (not raw observations)
        self.rnd = RNDModule(
            input_dim=self.clip.embed_dim,
            output_dim=rnd_output_dim,
            hidden_dim=rnd_hidden_dim,
            n_layers=rnd_n_layers,
        ).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations through CLIP."""
        return self.clip.encode_observation(obs)

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RND intrinsic reward on CLIP-encoded next observations."""
        with torch.no_grad():
            clip_features = self._encode(next_obs)
            reward = self.rnd.compute_intrinsic_reward(clip_features)

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)
        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train RND predictor on CLIP-encoded observations."""
        with torch.no_grad():
            clip_features = self._encode(batch["next_obs"])

        loss = self.rnd.compute_loss(clip_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()
        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {"rnd": self.rnd.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.rnd.load_state_dict(state["rnd"])
        self.optimizer.load_state_dict(state["optimizer"])
