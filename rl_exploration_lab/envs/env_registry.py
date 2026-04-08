"""Environment registry for standardized access to MiniGrid environments."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs with gymnasium

# Canonical benchmark environments (progressive difficulty)
BENCHMARK_ENVS: dict[str, dict] = {
    "Empty-8x8": {
        "id": "MiniGrid-Empty-8x8-v0",
        "difficulty": "easy",
        "description": "Sanity check — every method should solve this.",
    },
    "DoorKey-6x6": {
        "id": "MiniGrid-DoorKey-6x6-v0",
        "difficulty": "medium",
        "description": "Tests basic exploration + object interaction.",
    },
    "KeyCorridorS3R2": {
        "id": "MiniGrid-KeyCorridorS3R2-v0",
        "difficulty": "hard",
        "description": "Main thesis benchmark. Sparse reward, partial observability.",
    },
    "MultiRoomN6": {
        "id": "MiniGrid-MultiRoom-N6-v0",
        "difficulty": "hard",
        "description": "Tests long-horizon exploration across many rooms.",
    },
    "ObstructedMaze-1Dl": {
        "id": "MiniGrid-ObstructedMaze-1Dl-v0",
        "difficulty": "very_hard",
        "description": "Tests memory + exploration jointly.",
    },
}


def make_env(env_name: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create a MiniGrid environment by short name or full gymnasium ID.

    Args:
        env_name: Either a short name from BENCHMARK_ENVS (e.g. 'KeyCorridorS3R2')
                  or a full gymnasium ID (e.g. 'MiniGrid-KeyCorridorS3R2-v0').
        seed: Random seed for the environment.
        render_mode: Gymnasium render mode ('human', 'rgb_array', or None).

    Returns:
        A gymnasium environment instance.
    """
    if env_name in BENCHMARK_ENVS:
        env_id = BENCHMARK_ENVS[env_name]["id"]
    else:
        env_id = env_name

    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
    return env


def list_envs() -> list[str]:
    """Return list of registered benchmark environment short names."""
    return list(BENCHMARK_ENVS.keys())
