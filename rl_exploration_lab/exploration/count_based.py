"""Count-Based exploration (Bellemare et al., 2016).

Intrinsic reward is inversely proportional to the visitation count of each
state-action pair: r_i = beta / sqrt(N(s, a)).

States that haven't been visited often get higher exploration bonuses.
Simple but effective in tabular/low-dimensional environments like MiniGrid.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from rl_exploration_lab.exploration.base import BaseExploration


class CountBased(BaseExploration):
    """Count-Based exploration with visitation counts.

    Uses a hash of the observation as the state key. For MiniGrid's small
    observation space (7x7x3 = 147 dims), this is tractable.

    Args:
        beta: Scaling factor for the exploration bonus.
        device: Torch device.
    """

    def __init__(self, beta: float = 0.1, device: str = "cpu"):
        super().__init__(device=device)
        self.beta = beta
        self.visit_counts: dict[int, int] = defaultdict(int)
        self._total_updates = 0

    def _obs_to_key(self, obs: torch.Tensor) -> int:
        """Hash a single observation tensor to an integer key."""
        # Discretize to reduce near-duplicate states
        discretized = (obs * 255).byte()
        return hash(discretized.cpu().numpy().tobytes())

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute count-based intrinsic reward: beta / sqrt(N(s')).

        We reward based on the novelty of the *next* state (where we ended up).
        """
        batch_size = next_obs.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            key = self._obs_to_key(next_obs[i])
            self.visit_counts[key] += 1
            count = self.visit_counts[key]
            rewards[i] = self.beta / (count ** 0.5)

        return rewards

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """No trainable parameters — just report stats."""
        self._total_updates += 1
        return {
            "unique_states": len(self.visit_counts),
            "total_visits": sum(self.visit_counts.values()),
        }

    def state_dict(self) -> dict:
        return {"visit_counts": dict(self.visit_counts), "beta": self.beta}

    def load_state_dict(self, state: dict) -> None:
        self.visit_counts = defaultdict(int, state.get("visit_counts", {}))
        self.beta = state.get("beta", self.beta)
