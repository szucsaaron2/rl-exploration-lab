"""SHELM + RND + Language Oracle: thesis proposed improvement (Chapter 6).

The thesis found that CLIP's language encoder produces generic tokens like
"pixel", "pong", "square" for MiniGrid states — lacking task-specific
semantics like door colors or key positions.

This method replaces CLIP's implicit encoding with explicit ground-truth
language descriptions from the MiniGrid language oracle. The oracle provides
precise state descriptions that are then encoded by CLIP's text encoder
to produce the RND target.

From the thesis Discussion (§6):
    "We propose introducing a language oracle to generate precise text
     descriptions of MiniGrid states... This step could reveal whether
     our zero-reward outcomes stem from the encoder's misalignment with
     the environment or from other flaws."

This is the key experiment that can validate or invalidate the thesis hypothesis.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rl_exploration_lab.envs.language_oracle import LanguageOracle
from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.encoders import CLIPEncoder
from rl_exploration_lab.networks.predictors import RNDNetwork


class SHELMOracle(BaseExploration):
    """SHELM + RND with language oracle (thesis proposed fix).

    Instead of relying on CLIP's implicit language abstraction, uses
    ground-truth text descriptions from the language oracle. The oracle
    description is encoded through CLIP's text encoder to produce the
    RND target representation.

    If this method achieves non-zero rewards where SHELM-RND failed,
    it confirms the thesis hypothesis that CLIP's misalignment was the problem.

    Args:
        obs_dim: Raw observation dimension.
        clip_model: CLIP model name.
        predictor_hidden_dim: Predictor network hidden size.
        predictor_n_layers: Predictor network depth.
        lr: Predictor learning rate.
        oracle_verbose: Whether to use verbose oracle descriptions.
        reward_clip: Max intrinsic reward.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        clip_model: str = "ViT-B/32",
        predictor_hidden_dim: int = 128,
        predictor_n_layers: int = 2,
        lr: float = 1e-3,
        oracle_verbose: bool = True,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.reward_clip = reward_clip
        self._last_loss = 0.0
        self._last_descriptions: list[str] = []

        # Components
        self.clip = CLIPEncoder(model_name=clip_model, device=device)
        self.oracle = LanguageOracle(verbose=oracle_verbose)

        # Predictor: learns to predict oracle-based CLIP encoding from raw obs
        self.predictor = RNDNetwork(
            input_dim=obs_dim,
            output_dim=self.clip.embed_dim,
            hidden_dim=predictor_hidden_dim,
            n_layers=predictor_n_layers,
        ).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

    def _get_oracle_embeddings(self, obs: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """Generate oracle descriptions and encode them through CLIP.

        Args:
            obs: Flat observations, shape (batch, obs_dim).

        Returns:
            embeddings: CLIP text embeddings of oracle descriptions.
            descriptions: The raw text descriptions (for logging/interpretability).
        """
        descriptions = []
        for i in range(obs.shape[0]):
            obs_np = obs[i].cpu().numpy()
            desc = self.oracle.describe_observation(obs_np)
            descriptions.append(desc)

        embeddings = self.clip.encode_text(descriptions)
        return embeddings, descriptions

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intrinsic reward: prediction error vs oracle-based CLIP encoding.

        The oracle provides a ground-truth description of next_obs, which is
        encoded by CLIP's text encoder. The predictor tries to predict this
        encoding from the raw observation.
        """
        with torch.no_grad():
            # Oracle target: ground-truth description → CLIP text encoding
            oracle_target, descriptions = self._get_oracle_embeddings(next_obs)
            self._last_descriptions = descriptions

            # Predictor output
            predictor_out = self.predictor(next_obs)

            # Prediction error = intrinsic reward
            reward = (oracle_target - predictor_out).pow(2).mean(dim=-1)

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)
        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the predictor to match oracle-based CLIP encodings."""
        next_obs = batch["next_obs"]

        with torch.no_grad():
            oracle_target, _ = self._get_oracle_embeddings(next_obs)

        predictor_out = self.predictor(next_obs)
        loss = (oracle_target.detach() - predictor_out).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()
        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def get_last_descriptions(self) -> list[str]:
        """Return the most recent oracle descriptions (for interpretability)."""
        return self._last_descriptions

    def state_dict(self) -> dict:
        return {"predictor": self.predictor.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
