"""SHELM + RND: the thesis contribution (Szűcs, 2025).

Combines SHELM's human-readable language-based memory with RND's intrinsic
reward mechanism. SHELM's memory output serves as the RND target instead of
a randomly initialized frozen network.

From the thesis (§3.2):
    "We wanted to extend this lesson by using SHELM's human-readable,
     language-based memory with RND's intrinsic reward mechanism, to make
     exploration faster and more interpretable in POMDPs."

The prediction error between the predictor network and SHELM's memory
output provides the intrinsic reward signal.

Three variants tested in the thesis (§5.2):
- SHELM RND embeddings: top-k token embeddings as target
- SHELM RND single/2/4 decoded token strings: re-encoded text as target
"""

from __future__ import annotations

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.exploration.shelm.memory import SHELMMemory
from rl_exploration_lab.networks.encoders import CLIPEncoder
from rl_exploration_lab.networks.predictors import RNDNetwork


class SHELMRND(BaseExploration):
    """SHELM + RND exploration (thesis method).

    Uses SHELM's memory output as the RND target. The predictor learns
    to predict SHELM's language-based representation, and the prediction
    error serves as intrinsic reward.

    Args:
        obs_dim: Raw observation dimension.
        clip_model: CLIP model name.
        top_k: Number of tokens to retrieve from SHELM's semantic database.
        output_mode: SHELM output mode ('embeddings', 'tokens', 'average').
        predictor_hidden_dim: Predictor network hidden size.
        predictor_n_layers: Predictor network depth.
        lr: Predictor learning rate.
        reward_clip: Max intrinsic reward.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        clip_model: str = "ViT-B/32",
        top_k: int = 4,
        output_mode: str = "embeddings",
        predictor_hidden_dim: int = 128,
        predictor_n_layers: int = 2,
        lr: float = 1e-3,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.reward_clip = reward_clip
        self._last_loss = 0.0
        self._last_tokens: list[list[str]] = []

        # SHELM memory (provides the target)
        self.clip = CLIPEncoder(model_name=clip_model, device=device)
        self.shelm = SHELMMemory(self.clip, top_k=top_k, output_mode=output_mode)

        # Predictor network (learns to predict SHELM's output from raw obs)
        target_dim = self.shelm.output_dim
        self.predictor = RNDNetwork(
            input_dim=obs_dim,
            output_dim=target_dim,
            hidden_dim=predictor_hidden_dim,
            n_layers=predictor_n_layers,
        ).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SHELM-RND intrinsic reward.

        Intrinsic reward = MSE between predictor output and SHELM memory output
        for the next observation.
        """
        with torch.no_grad():
            # SHELM target: language-based memory representation
            shelm_target, tokens = self.shelm(next_obs)
            self._last_tokens = tokens

            # Predictor: learns to predict SHELM's output from raw obs
            predictor_out = self.predictor(next_obs)

            # Prediction error = intrinsic reward
            reward = (shelm_target - predictor_out).pow(2).mean(dim=-1)

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)
        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the predictor to match SHELM's memory output."""
        next_obs = batch["next_obs"]

        with torch.no_grad():
            shelm_target, _ = self.shelm(next_obs)

        predictor_out = self.predictor(next_obs)
        loss = (shelm_target.detach() - predictor_out).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()
        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def get_last_tokens(self) -> list[list[str]]:
        """Return the most recently retrieved tokens (for interpretability logging)."""
        return self._last_tokens

    def state_dict(self) -> dict:
        return {"predictor": self.predictor.state_dict(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.predictor.load_state_dict(state["predictor"])
        self.optimizer.load_state_dict(state["optimizer"])
