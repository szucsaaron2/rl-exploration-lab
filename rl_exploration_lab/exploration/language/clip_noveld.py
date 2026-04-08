"""CLIP-NovelD: NovelD with CLIP-encoded observations.

Applies NovelD's novelty-difference mechanism to CLIP embeddings instead
of raw observations. Novelty is measured by RND prediction error in
CLIP embedding space.

r_i = max[novelty_CLIP(s') - α * novelty_CLIP(s), 0]

From the thesis (§5.1): CLIP NovelD was one of the best-performing baselines.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.encoders import CLIPEncoder
from rl_exploration_lab.networks.predictors import RNDModule


class CLIPNovelD(BaseExploration):
    """NovelD exploration using CLIP-encoded observations.

    Args:
        obs_dim: Raw observation dimension.
        clip_model: CLIP model name.
        rnd_output_dim: RND embedding output dimension.
        rnd_hidden_dim: RND hidden layer size.
        lr: Predictor learning rate.
        alpha: Novelty difference scaling factor.
        use_erir: Whether to use Episodic Restriction on Intrinsic Reward.
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
        alpha: float = 0.5,
        use_erir: bool = True,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.alpha = alpha
        self.use_erir = use_erir
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        self.clip = CLIPEncoder(model_name=clip_model, device=device)
        self.rnd = RNDModule(
            input_dim=self.clip.embed_dim, output_dim=rnd_output_dim,
            hidden_dim=rnd_hidden_dim, n_layers=2,
        ).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        self._episodic_visited: set[int] = set()

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_observation(obs)

    def _novelty(self, clip_features: torch.Tensor) -> torch.Tensor:
        return self.rnd.compute_intrinsic_reward(clip_features)

    def _obs_key(self, clip_features: torch.Tensor) -> int:
        discretized = (clip_features * 100).int()
        return hash(discretized.cpu().numpy().tobytes())

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            clip_s = self._encode(obs)
            clip_s_next = self._encode(next_obs)

            novelty_current = self._novelty(clip_s)
            novelty_next = self._novelty(clip_s_next)

            reward = torch.clamp(novelty_next - self.alpha * novelty_current, min=0.0)

            if self.use_erir:
                for i in range(next_obs.shape[0]):
                    key = self._obs_key(clip_s_next[i])
                    if key in self._episodic_visited:
                        reward[i] = 0.0
                    else:
                        self._episodic_visited.add(key)

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
        self._episodic_visited.clear()
        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {"rnd": self.rnd.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.rnd.load_state_dict(state["rnd"])
        self.optimizer.load_state_dict(state["optimizer"])
