"""Rewarding Impact-Driven Exploration (Raileanu et al., 2020).

Rewards the agent for taking actions that cause significant changes in a
learned state representation, divided by episodic visit count to discourage
revisiting:

    r_i = |φ(s_{t+1}) - φ(s_t)|_2 / sqrt(N_ep(s_{t+1}))

Like ICM, RIDE uses forward and inverse dynamics models to learn the state
embedding φ. The key difference is that RIDE rewards *impact* (change in
representation) rather than *surprise* (prediction error).

Works well in procedurally generated environments like MiniGrid.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.dynamics import DynamicsModel


class RIDE(BaseExploration):
    """RIDE: Rewarding Impact-Driven Exploration.

    Args:
        obs_dim: Observation dimension.
        n_actions: Number of discrete actions.
        embed_dim: Feature embedding dimension.
        hidden_dim: Hidden layer size.
        lr: Learning rate for the dynamics model.
        reward_clip: Maximum intrinsic reward. None for no clipping.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        n_actions: int = 7,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        self.dynamics = DynamicsModel(obs_dim, n_actions, embed_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.dynamics.parameters(), lr=lr)

        # Episodic visit counts (reset each episode)
        self._episodic_counts: dict[int, int] = defaultdict(int)
        self._in_episode = True

    def _state_key(self, phi: torch.Tensor) -> int:
        """Hash a state embedding for episodic counting."""
        # Discretize the embedding for hashing
        discretized = (phi * 100).int()
        return hash(discretized.cpu().numpy().tobytes())

    def reset_episode(self):
        """Reset episodic visit counts at the start of a new episode."""
        self._episodic_counts.clear()

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute RIDE intrinsic reward: ||φ(s') - φ(s)||_2 / sqrt(N_ep(s'))."""
        batch_size = obs.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)

        with torch.no_grad():
            phi_s = self.dynamics.encode(obs)
            phi_s_next = self.dynamics.encode(next_obs)

            # Impact: L2 distance in embedding space
            impact = (phi_s_next - phi_s).pow(2).sum(dim=-1).sqrt()

            for i in range(batch_size):
                key = self._state_key(phi_s_next[i])
                self._episodic_counts[key] += 1
                count = self._episodic_counts[key]

                # Divide by sqrt of episodic count to discourage revisiting
                rewards[i] = impact[i] / (count ** 0.5)

        if self.reward_clip is not None:
            rewards = rewards.clamp(0.0, self.reward_clip)

        return rewards

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the dynamics model (forward + inverse) on the rollout batch."""
        obs = batch["obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]

        forward_loss, inverse_loss, _ = self.dynamics.compute_icm_losses(obs, next_obs, actions)
        total_loss = 0.2 * forward_loss + 0.8 * inverse_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._last_loss = total_loss.item()

        # Reset episodic counts after each update (approximation:
        # ideally reset per episode, but this is close enough for rollout-based training)
        self._episodic_counts.clear()

        return {
            "forward_loss": forward_loss.item(),
            "inverse_loss": inverse_loss.item(),
            "exploration_loss": self._last_loss,
        }

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {
            "dynamics": self.dynamics.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.dynamics.load_state_dict(state["dynamics"])
        self.optimizer.load_state_dict(state["optimizer"])
