"""NovelD: A Simple yet Effective Exploration Criterion (Zhang et al., 2021).

Rewards novelty *difference* between consecutive states rather than absolute novelty.
This pushes the agent toward the frontier of explored territory:

    r_i = max[novelty(s_{t+1}) - α * novelty(s_t), 0]

where novelty is measured by RND prediction error. The agent is only rewarded
when it moves to a *more* novel state than where it currently is.

Also includes Episodic Restriction on Intrinsic Reward (ERIR): the agent is
only rewarded for visiting a state for the first time in each episode.

NovelD outperforms RND and RIDE on MiniGrid, solving all environments within
120M steps — previous methods only solved half.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.predictors import RNDModule


class NovelD(BaseExploration):
    """NovelD exploration with ERIR.

    Args:
        obs_dim: Observation dimension.
        output_dim: RND embedding dimension.
        hidden_dim: Hidden layer size.
        n_layers: Number of hidden layers per RND network.
        lr: Learning rate for the RND predictor.
        alpha: Scaling factor for current-state novelty subtraction.
        use_erir: Whether to use Episodic Restriction on Intrinsic Reward.
        reward_clip: Maximum intrinsic reward. None for no clipping.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        output_dim: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 2,
        lr: float = 1e-3,
        alpha: float = 0.5,
        use_erir: bool = True,
        reward_clip: float | None = 1.0,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.alpha = alpha
        self.use_erir = use_erir
        self.reward_clip = reward_clip
        self._last_loss = 0.0

        self.rnd = RNDModule(obs_dim, output_dim, hidden_dim, n_layers).to(device)
        self.optimizer = optim.Adam(self.rnd.predictor.parameters(), lr=lr)

        # ERIR: track first visits per episode
        self._episodic_visited: set[int] = set()

    def _obs_key(self, obs: torch.Tensor) -> int:
        """Hash observation for ERIR tracking."""
        discretized = (obs * 255).byte()
        return hash(discretized.cpu().numpy().tobytes())

    def _novelty(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute per-sample novelty using RND prediction error."""
        return self.rnd.compute_intrinsic_reward(obs)

    def reset_episode(self):
        """Reset episodic visit tracking (call at episode start)."""
        self._episodic_visited.clear()

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NovelD intrinsic reward: max[novelty(s') - α*novelty(s), 0].

        With ERIR: zero reward if s' was already visited in this episode.
        """
        with torch.no_grad():
            novelty_current = self._novelty(obs)
            novelty_next = self._novelty(next_obs)

            # Novelty difference
            reward = torch.clamp(novelty_next - self.alpha * novelty_current, min=0.0)

            # ERIR: zero out reward for already-visited states
            if self.use_erir:
                batch_size = next_obs.shape[0]
                for i in range(batch_size):
                    key = self._obs_key(next_obs[i])
                    if key in self._episodic_visited:
                        reward[i] = 0.0
                    else:
                        self._episodic_visited.add(key)

        if self.reward_clip is not None:
            reward = reward.clamp(0.0, self.reward_clip)

        return reward

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Train the RND predictor on next observations."""
        next_obs = batch["next_obs"]

        loss = self.rnd.compute_loss(next_obs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._last_loss = loss.item()

        # Reset ERIR tracking after each rollout update
        self._episodic_visited.clear()

        return {"exploration_loss": self._last_loss}

    def get_exploration_loss(self) -> float | None:
        return self._last_loss

    def state_dict(self) -> dict:
        return {
            "rnd": self.rnd.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.rnd.load_state_dict(state["rnd"])
        self.optimizer.load_state_dict(state["optimizer"])
