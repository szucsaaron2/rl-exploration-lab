"""Never Give Up (Badia et al., 2020).

Combines episodic and life-long novelty for robust exploration in
sparse-reward environments.

    r_int = r_episodic * min(max(α_t, 1), L)

Episodic novelty: uses an episodic memory buffer with k-nearest-neighbors
to measure how novel the current state is *within the current episode*:

    r_episodic = 1 / sqrt(N_ep(f(s_t)))

Life-long novelty: uses RND prediction error to measure novelty across
the agent's entire lifetime (like standard RND).

The combination ensures the agent both explores new territory within each
episode (episodic) and across episodes (life-long).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.predictors import RNDModule


class NGU(BaseExploration):
    """Never Give Up exploration with episodic + life-long novelty.

    Simplified implementation:
    - Episodic novelty via hash-based visit counting (instead of full k-NN)
    - Life-long novelty via RND prediction error
    - Combined via multiplicative interaction

    Args:
        obs_dim: Observation dimension.
        output_dim: RND embedding dimension.
        hidden_dim: Hidden layer size.
        n_layers: Number of hidden layers for RND networks.
        lr: Learning rate for the RND predictor.
        max_reward_scale: Maximum reward scaling L.
        reward_clip: Clip final intrinsic reward. None for no clipping.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        output_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        lr: float = 1e-3,
        max_reward_scale: float = 5.0,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.max_reward_scale = max_reward_scale
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        # Life-long novelty via RND
        self.rnd = RNDModule(obs_dim, output_dim, hidden_dim, n_layers).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # Episodic novelty via visit counting
        self._episodic_counts: dict[int, int] = defaultdict(int)

        # Running statistics for life-long novelty normalization
        self._lifelong_running_mean = 0.0
        self._lifelong_running_var = 1.0
        self._lifelong_count = 0

    def _state_key(self, obs: torch.Tensor) -> int:
        """Hash observation for episodic counting."""
        discretized = (obs * 255).byte()
        return hash(discretized.cpu().numpy().tobytes())

    def _update_running_stats(self, values: torch.Tensor):
        """Update running mean/var for life-long novelty normalization."""
        batch_mean = values.mean().item()
        batch_var = values.var().item() if values.numel() > 1 else 0.0
        batch_count = values.numel()

        total_count = self._lifelong_count + batch_count
        if total_count == 0:
            return

        delta = batch_mean - self._lifelong_running_mean
        new_mean = self._lifelong_running_mean + delta * batch_count / total_count
        m_a = self._lifelong_running_var * self._lifelong_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self._lifelong_count * batch_count / total_count
        new_var = m2 / total_count

        self._lifelong_running_mean = new_mean
        self._lifelong_running_var = max(new_var, 1e-8)
        self._lifelong_count = total_count

    def reset_episode(self):
        """Reset episodic memory at the start of a new episode."""
        self._episodic_counts.clear()

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NGU intrinsic reward: r_episodic * clamp(α_t, 1, L).

        r_episodic = 1 / sqrt(N_ep(s'))
        α_t = normalized RND error (life-long novelty)
        """
        batch_size = next_obs.shape[0]

        with torch.no_grad():
            # Life-long novelty via RND
            lifelong_novelty = self.rnd.compute_intrinsic_reward(next_obs)
            self._update_running_stats(lifelong_novelty)

            # Normalize life-long novelty
            alpha = (lifelong_novelty - self._lifelong_running_mean) / (
                self._lifelong_running_var**0.5 + 1e-8
            )

            # Episodic novelty
            episodic_reward = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                key = self._state_key(next_obs[i])
                self._episodic_counts[key] += 1
                count = self._episodic_counts[key]
                episodic_reward[i] = 1.0 / (count**0.5)

            # Combined: r_int = r_episodic * clamp(max(alpha, 1), 1, L)
            modulated_alpha = torch.clamp(
                torch.clamp(alpha, min=1.0), max=self.max_reward_scale
            )
            intrinsic_reward = episodic_reward * modulated_alpha

        if self.reward_clip is not None:
            intrinsic_reward = intrinsic_reward.clamp(0.0, self.reward_clip)

        return intrinsic_reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the RND predictor (life-long novelty component)."""
        next_obs = batch["next_obs"]

        loss = self.rnd.compute_loss(next_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()

        # Reset episodic counts after each update
        self._episodic_counts.clear()

        return {
            "exploration_loss": self._last_loss,
            "episodic_unique_states": len(self._episodic_counts),
            "lifelong_mean_novelty": self._lifelong_running_mean,
        }

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {
            "rnd": self.rnd.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lifelong_mean": self._lifelong_running_mean,
            "lifelong_var": self._lifelong_running_var,
            "lifelong_count": self._lifelong_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.rnd.load_state_dict(state["rnd"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._lifelong_running_mean = state.get("lifelong_mean", 0.0)
        self._lifelong_running_var = state.get("lifelong_var", 1.0)
        self._lifelong_count = state.get("lifelong_count", 0)
