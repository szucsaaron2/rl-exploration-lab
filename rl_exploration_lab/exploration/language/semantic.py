"""Semantic Exploration (Tam et al., 2022).

Uses a vision-language model (CLIP) to encourage exploration of
semantically different states. Built on NGU (episodic + lifelong novelty)
but with CLIP embeddings replacing raw visual features.

Key insight from the paper: language embeddings capture subtle semantic
differences better than raw visual features. Both oracle-based and
CLIP-based versions performed well, but the oracle version was better,
suggesting room for improvement in visual-language alignment.

This motivated the SHELM approach in the thesis.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.encoders import CLIPEncoder
from rl_exploration_lab.networks.predictors import RNDModule


class SemanticExploration(BaseExploration):
    """Semantic exploration using CLIP embeddings with NGU-style rewards.

    Combines:
    - Episodic novelty: visit counting in CLIP embedding space
    - Life-long novelty: RND prediction error on CLIP embeddings

    Args:
        obs_dim: Raw observation dimension.
        clip_model: CLIP model name.
        rnd_output_dim: RND embedding output dimension.
        rnd_hidden_dim: RND hidden layer size.
        lr: Predictor learning rate.
        max_reward_scale: NGU max reward scaling L.
        reward_clip: Max intrinsic reward.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        clip_model: str = "ViT-B/32",
        rnd_output_dim: int = 64,
        rnd_hidden_dim: int = 128,
        lr: float = 1e-3,
        max_reward_scale: float = 5.0,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.max_reward_scale = max_reward_scale
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        self.clip = CLIPEncoder(model_name=clip_model, device=device)

        # RND on CLIP embeddings (life-long novelty)
        self.rnd = RNDModule(
            input_dim=self.clip.embed_dim, output_dim=rnd_output_dim,
            hidden_dim=rnd_hidden_dim, n_layers=2,
        ).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # Episodic counting in CLIP space
        self._episodic_counts: dict[int, int] = defaultdict(int)
        self._lifelong_mean = 0.0
        self._lifelong_var = 1.0
        self._lifelong_count = 0

    def _state_key(self, clip_emb: torch.Tensor) -> int:
        discretized = (clip_emb * 100).int()
        return hash(discretized.cpu().numpy().tobytes())

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_observation(obs)

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = next_obs.shape[0]

        with torch.no_grad():
            clip_features = self._encode(next_obs)

            # Life-long novelty via RND on CLIP embeddings
            lifelong = self.rnd.compute_intrinsic_reward(clip_features)

            # Update running stats
            batch_mean = lifelong.mean().item()
            self._lifelong_count += batch_size
            self._lifelong_mean += (
                (batch_mean - self._lifelong_mean) * batch_size / self._lifelong_count
            )

            # Normalize
            alpha = (lifelong - self._lifelong_mean) / (self._lifelong_var ** 0.5 + 1e-8)

            # Episodic novelty in CLIP space
            episodic = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                key = self._state_key(clip_features[i])
                self._episodic_counts[key] += 1
                episodic[i] = 1.0 / (self._episodic_counts[key] ** 0.5)

            # NGU-style combination
            modulated = torch.clamp(torch.clamp(alpha, min=1.0), max=self.max_reward_scale)
            reward = episodic * modulated

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)
        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.no_grad():
            clip_features = self._encode(batch["next_obs"])

        loss = self.rnd.compute_loss(clip_features)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()
        self._episodic_counts.clear()
        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {"rnd": self.rnd.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.rnd.load_state_dict(state["rnd"])
        self.optimizer.load_state_dict(state["optimizer"])
