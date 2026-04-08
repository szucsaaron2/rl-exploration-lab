"""MiniGrid environment wrapper providing standardized observation and action spaces.

Wraps MiniGrid environments to provide:
- Flat tensor observations from the 7x7x3 partial grid view
- Full grid state extraction for coverage metrics and Go-Explore cell representations
- Episode statistics tracking (reward, length, solved)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch


class MiniGridWrapper(gym.Wrapper):
    """Standardized wrapper for MiniGrid environments.

    Converts the dict observation space to a flat numpy array and tracks
    episode statistics for evaluation.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # MiniGrid observations are dicts with 'image' (7x7x3), 'direction', 'mission'
        # We flatten the image to a 1D vector for neural network input
        self.obs_shape = (7 * 7 * 3,)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.float32
        )
        # MiniGrid has 7 discrete actions
        self.n_actions = self.action_space.n

        # Episode tracking
        self._episode_reward = 0.0
        self._episode_length = 0
        self._episode_count = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._episode_reward = 0.0
        self._episode_length = 0
        return self._process_obs(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward
        self._episode_length += 1

        if terminated or truncated:
            self._episode_count += 1
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
                "solved": reward > 0,
            }

        return self._process_obs(obs), reward, terminated, truncated, info

    def _process_obs(self, obs: dict) -> np.ndarray:
        """Flatten the 7x7x3 grid observation to a 1D float array."""
        image = obs["image"]  # shape (7, 7, 3)
        return image.flatten().astype(np.float32) / 255.0

    def get_full_grid(self) -> np.ndarray:
        """Get the full environment grid (not just the agent's view).

        Useful for coverage metrics and Go-Explore cell representations.
        Returns the full grid encoding as a numpy array.
        """
        grid = self.env.unwrapped.grid.encode()
        return grid

    def get_agent_state(self) -> dict:
        """Extract the agent's current state for Go-Explore cell representations.

        Returns dict with agent position, direction, and carried object.
        """
        env = self.env.unwrapped
        return {
            "pos": tuple(env.agent_pos),
            "dir": env.agent_dir,
            "carrying": str(env.carrying) if env.carrying else None,
        }

    def get_obs_tensor(self, obs: np.ndarray, device: str = "cpu") -> torch.Tensor:
        """Convert a numpy observation to a torch tensor."""
        return torch.from_numpy(obs).float().to(device)

    @property
    def episode_count(self) -> int:
        return self._episode_count


def make_wrapped_env(
    env_name: str, seed: int | None = None, render_mode: str | None = None
) -> MiniGridWrapper:
    """Create a wrapped MiniGrid environment.

    Args:
        env_name: Short name or full gymnasium ID.
        seed: Random seed.
        render_mode: Gymnasium render mode.

    Returns:
        Wrapped MiniGrid environment.
    """
    from rl_exploration_lab.envs.env_registry import make_env

    env = make_env(env_name, seed=seed, render_mode=render_mode)
    return MiniGridWrapper(env)
