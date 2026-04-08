"""L-NovelD: Language-conditioned NovelD (Li et al., 2022).

Modifies NovelD by defining novelty based on language descriptions of states
instead of raw observations. Uses a language oracle L(s) to get ground-truth
text descriptions, then applies RND on the language encoding.

novelty(s) = RND_error(CLIP_text(L(s)))
r_i = max[novelty(s') - α * novelty(s), 0]

From the thesis (§2.2.10): "L-NovelD augments NovelD by incorporating a
language-based novelty measure."
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rl_exploration_lab.envs.language_oracle import LanguageOracle
from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.encoders import CLIPEncoder
from rl_exploration_lab.networks.predictors import RNDModule


class LNovelD(BaseExploration):
    """Language-conditioned NovelD.

    Args:
        obs_dim: Raw observation dimension.
        clip_model: CLIP model name.
        rnd_output_dim: RND embedding dimension.
        rnd_hidden_dim: RND hidden size.
        lr: RND predictor learning rate.
        alpha: NovelD scaling factor.
        use_erir: Episodic Restriction on Intrinsic Reward.
        oracle_verbose: Whether to use verbose oracle descriptions.
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
        oracle_verbose: bool = False,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.alpha = alpha
        self.use_erir = use_erir
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        self.clip = CLIPEncoder(model_name=clip_model, device=device)
        self.oracle = LanguageOracle(verbose=oracle_verbose)

        # RND on language-encoded state descriptions
        self.rnd = RNDModule(
            input_dim=self.clip.embed_dim, output_dim=rnd_output_dim,
            hidden_dim=rnd_hidden_dim, n_layers=2,
        ).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        self._episodic_visited: set[str] = set()

    def _get_language_encoding(self, obs: torch.Tensor) -> torch.Tensor:
        """Get CLIP text encoding of oracle descriptions."""
        descriptions = []
        for i in range(obs.shape[0]):
            desc = self.oracle.describe_observation(obs[i].cpu().numpy())
            descriptions.append(desc)
        return self.clip.encode_text(descriptions)

    def _novelty(self, lang_features: torch.Tensor) -> torch.Tensor:
        return self.rnd.compute_intrinsic_reward(lang_features)

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            lang_s = self._get_language_encoding(obs)
            lang_s_next = self._get_language_encoding(next_obs)

            novelty_current = self._novelty(lang_s)
            novelty_next = self._novelty(lang_s_next)

            reward = torch.clamp(novelty_next - self.alpha * novelty_current, min=0.0)

            if self.use_erir:
                for i in range(next_obs.shape[0]):
                    desc = self.oracle.describe_observation(next_obs[i].cpu().numpy())
                    if desc in self._episodic_visited:
                        reward[i] = 0.0
                    else:
                        self._episodic_visited.add(desc)

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)
        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        with torch.no_grad():
            lang_features = self._get_language_encoding(batch["next_obs"])

        loss = self.rnd.compute_loss(lang_features)
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
