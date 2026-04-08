"""Predictor and target networks for RND-family exploration methods.

Provides the fixed target network and trainable predictor network
used by RND, NovelD, NGU, CLIP-RND, and SHELM variants.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RNDNetwork(nn.Module):
    """A simple MLP used as either target or predictor in RND.

    Args:
        input_dim: Dimension of the input (observation embedding).
        output_dim: Dimension of the output embedding.
        hidden_dim: Hidden layer size.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        input_dim: int = 147,
        output_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDModule(nn.Module):
    """Complete RND module with frozen target and trainable predictor.

    The intrinsic reward is the MSE between the target and predictor outputs.
    States that haven't been seen before produce high prediction error (novelty).

    Args:
        input_dim: Observation dimension.
        output_dim: Embedding dimension for target/predictor outputs.
        hidden_dim: Hidden layer size.
        n_layers: Number of hidden layers per network.
    """

    def __init__(
        self,
        input_dim: int = 147,
        output_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ):
        super().__init__()
        self.target = RNDNetwork(input_dim, output_dim, hidden_dim, n_layers)
        self.predictor = RNDNetwork(input_dim, output_dim, hidden_dim, n_layers)

        # Freeze the target network — its weights are random and fixed
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute target and predictor outputs.

        Args:
            obs: Batch of observations, shape (batch, input_dim).

        Returns:
            target_out: Fixed target network output.
            predictor_out: Trainable predictor network output.
        """
        with torch.no_grad():
            target_out = self.target(obs)
        predictor_out = self.predictor(obs)
        return target_out, predictor_out

    def compute_intrinsic_reward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute RND intrinsic reward (prediction error).

        Args:
            obs: Batch of observations, shape (batch, input_dim).

        Returns:
            Intrinsic rewards, shape (batch,).
        """
        target_out, predictor_out = self(obs)
        # MSE per sample (reduced over output dim, kept over batch)
        intrinsic_reward = (target_out - predictor_out).pow(2).mean(dim=-1)
        return intrinsic_reward

    def compute_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the predictor's training loss (MSE against target).

        Args:
            obs: Batch of observations.

        Returns:
            Scalar MSE loss.
        """
        target_out, predictor_out = self(obs)
        return (target_out - predictor_out).pow(2).mean()
