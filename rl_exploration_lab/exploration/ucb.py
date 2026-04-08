"""Upper Confidence Bound exploration adapted for RL (Lattimore & Szepesvári, 2020).

In classic UCB, action selection balances estimated reward with an uncertainty
bonus: a_t = argmax_a [Q(a) + c * sqrt(log(t) / N(a))].

For deep RL integration, we use UCB as an intrinsic reward bonus based on
state-action visitation counts, similar to count-based but with the UCB
exploration term that accounts for global step count.

r_i = c * sqrt(log(t) / N(s, a))
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from rl_exploration_lab.exploration.base import BaseExploration


class UCB(BaseExploration):
    """UCB-style exploration bonus for deep RL.

    Provides an intrinsic reward proportional to the UCB exploration term.
    Uses observation hashing for state identification (same as CountBased).

    Args:
        c: Exploration constant controlling the bonus magnitude.
        device: Torch device.
    """

    def __init__(self, c: float = 1.0, device: str = "cpu"):
        super().__init__(device=device)
        self.c = c
        self.visit_counts: dict[int, int] = defaultdict(int)
        self._total_steps = 0

    def _obs_action_key(self, obs: torch.Tensor, action: int) -> int:
        """Hash observation + action to a unique key."""
        discretized = (obs * 255).byte()
        return hash((discretized.cpu().numpy().tobytes(), action))

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute UCB intrinsic reward: c * sqrt(log(t) / N(s, a))."""
        batch_size = obs.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            self._total_steps += 1
            key = self._obs_action_key(obs[i], action[i].item())
            self.visit_counts[key] += 1
            count = self.visit_counts[key]

            if self._total_steps > 1 and count > 0:
                rewards[i] = self.c * math.sqrt(math.log(self._total_steps) / count)

        return rewards

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            "unique_state_actions": len(self.visit_counts),
            "total_steps": self._total_steps,
        }

    def state_dict(self) -> dict:
        return {
            "visit_counts": dict(self.visit_counts),
            "total_steps": self._total_steps,
            "c": self.c,
        }

    def load_state_dict(self, state: dict) -> None:
        self.visit_counts = defaultdict(int, state.get("visit_counts", {}))
        self._total_steps = state.get("total_steps", 0)
        self.c = state.get("c", self.c)
