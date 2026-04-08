"""Actor-Critic policy networks for PPO.

Provides CNN and MLP-based policy networks that output both action logits
and value estimates from MiniGrid observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ObsEncoder(nn.Module):
    """Encode flat MiniGrid observations into a feature vector.

    Takes the flattened 7x7x3 = 147 observation and produces a dense embedding.
    Uses a simple MLP (MiniGrid observations are small enough that CNNs aren't needed,
    though a CNN variant is provided for extensibility).
    """

    def __init__(self, obs_dim: int = 147, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Shared encoder with separate actor (policy) and critic (value) heads.

    Args:
        obs_dim: Dimension of the flat observation vector.
        n_actions: Number of discrete actions.
        hidden_dim: Hidden layer size for the encoder.
        embed_dim: Embedding dimension from the encoder.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        n_actions: int = 7,
        hidden_dim: int = 128,
        embed_dim: int = 64,
    ):
        super().__init__()
        self.encoder = ObsEncoder(obs_dim, hidden_dim, embed_dim)

        # Actor head: outputs action logits
        self.actor = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Critic head: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2**0.5)
                nn.init.zeros_(module.bias)
        # Smaller init for policy output layer (encourages initial exploration)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        # Smaller init for value output layer
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        """Forward pass returning action distribution and value estimate.

        Args:
            obs: Batch of observations, shape (batch, obs_dim).

        Returns:
            dist: Categorical distribution over actions.
            value: State value estimates, shape (batch, 1).
        """
        features = self.encoder(obs)
        logits = self.actor(features)
        value = self.critic(features)
        dist = Categorical(logits=logits)
        return dist, value

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value for PPO update.

        Args:
            obs: Batch of observations.
            action: If provided, compute log_prob for these actions (for PPO update).
                    If None, sample new actions (for rollout collection).

        Returns:
            action: Sampled or provided actions.
            log_prob: Log probability of the actions.
            entropy: Entropy of the action distribution.
            value: State value estimates.
        """
        dist, value = self(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)
