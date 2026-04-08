"""Forward and inverse dynamics models for curiosity-driven exploration (ICM, RIDE).

The inverse model predicts the action from consecutive state encodings.
The forward model predicts the next state encoding from current state + action.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """Combined forward and inverse dynamics model for ICM / RIDE.

    Args:
        obs_dim: Observation dimension (flat).
        n_actions: Number of discrete actions.
        embed_dim: Embedding dimension for state features.
        hidden_dim: Hidden layer size.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        n_actions: int = 7,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.embed_dim = embed_dim

        # State encoder (shared between forward and inverse models)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Inverse dynamics: predict action from (phi(s), phi(s'))
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Forward dynamics: predict phi(s') from (phi(s), action)
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation into feature space."""
        return self.encoder(obs)

    def predict_action(
        self, phi_s: torch.Tensor, phi_s_next: torch.Tensor
    ) -> torch.Tensor:
        """Inverse model: predict action from consecutive state features.

        Returns logits over actions.
        """
        combined = torch.cat([phi_s, phi_s_next], dim=-1)
        return self.inverse_model(combined)

    def predict_next_state(
        self, phi_s: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Forward model: predict next state features from current state + action.

        Returns predicted phi(s').
        """
        action_onehot = F.one_hot(action.long(), self.n_actions).float()
        combined = torch.cat([phi_s, action_onehot], dim=-1)
        return self.forward_model(combined)

    def compute_icm_losses(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute ICM forward loss, inverse loss, and intrinsic reward.

        Args:
            obs: Current observations, shape (batch, obs_dim).
            next_obs: Next observations, shape (batch, obs_dim).
            action: Actions taken, shape (batch,).

        Returns:
            forward_loss: MSE between predicted and actual next state features.
            inverse_loss: Cross-entropy loss for action prediction.
            intrinsic_reward: Per-sample forward prediction error, shape (batch,).
        """
        phi_s = self.encode(obs)
        phi_s_next = self.encode(next_obs)

        # Inverse model loss
        action_logits = self.predict_action(phi_s, phi_s_next)
        inverse_loss = F.cross_entropy(action_logits, action.long())

        # Forward model loss and intrinsic reward
        phi_s_next_pred = self.predict_next_state(phi_s, action)
        forward_loss = F.mse_loss(phi_s_next_pred, phi_s_next.detach())

        # Intrinsic reward: per-sample prediction error
        intrinsic_reward = (phi_s_next_pred - phi_s_next.detach()).pow(2).mean(dim=-1)

        return forward_loss, inverse_loss, intrinsic_reward
