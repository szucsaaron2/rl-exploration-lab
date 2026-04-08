"""Go-Explore: Phase 1 — Explore until solved (Ecoffet et al., 2019/2021).

The main exploration loop:
1. Select a cell from the archive
2. Return to that cell (restore simulator state — no exploration noise)
3. Explore from that cell with random actions for K steps
4. Add any newly discovered cells to the archive
5. Repeat

No neural network is used in Phase 1 — exploration is purely random from
good stepping stones. This is surprisingly effective.

For MiniGrid, we exploit determinism by saving/restoring simulator states.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from tqdm import tqdm

from rl_exploration_lab.envs.minigrid_wrapper import MiniGridWrapper
from rl_exploration_lab.exploration.go_explore.archive import Archive, CellEntry
from rl_exploration_lab.exploration.go_explore.cell_repr import (
    Cell,
    CellRepresentation,
    MiniGridDomainCell,
)


class GoExplorePhase1:
    """Go-Explore Phase 1: archive-based exploration.

    Args:
        env: Wrapped MiniGrid environment.
        cell_repr: Cell representation strategy.
        explore_steps: Number of random action steps per exploration round.
        action_repeat_prob: Probability of repeating the previous action (sticky actions).
        total_steps: Maximum total environment steps for Phase 1.
    """

    def __init__(
        self,
        env: MiniGridWrapper,
        cell_repr: CellRepresentation | None = None,
        explore_steps: int = 100,
        action_repeat_prob: float = 0.95,
        total_steps: int = 1_000_000,
    ):
        self.env = env
        self.cell_repr = cell_repr or MiniGridDomainCell()
        self.explore_steps = explore_steps
        self.action_repeat_prob = action_repeat_prob
        self.total_steps = total_steps

        self.archive = Archive()
        self.n_actions = env.n_actions
        self._global_step = 0

        # Stats
        self._best_score = 0.0
        self._episodes_completed = 0

    def _get_env_state(self) -> Any:
        """Save the full simulator state for later restoration."""
        inner_env = self.env.env.unwrapped
        # Use deepcopy to capture the full MiniGrid state
        return {
            "grid": copy.deepcopy(inner_env.grid),
            "agent_pos": tuple(inner_env.agent_pos),
            "agent_dir": inner_env.agent_dir,
            "carrying": copy.deepcopy(inner_env.carrying),
            "step_count": inner_env.step_count,
        }

    def _restore_env_state(self, state: Any) -> np.ndarray:
        """Restore the simulator to a previously saved state.

        Returns the observation at the restored state.
        """
        inner_env = self.env.env.unwrapped
        inner_env.grid = copy.deepcopy(state["grid"])
        inner_env.agent_pos = np.array(state["agent_pos"])
        inner_env.agent_dir = state["agent_dir"]
        inner_env.carrying = copy.deepcopy(state["carrying"])
        inner_env.step_count = state["step_count"]

        # Generate observation from restored state
        obs = inner_env.gen_obs()
        return self.env._process_obs(obs)

    def _get_cell(self, obs: np.ndarray) -> Cell:
        """Get the cell for the current observation + agent state."""
        agent_state = self.env.get_agent_state()
        return self.cell_repr.obs_to_cell(obs, env_state=agent_state)

    def _explore_from(self, entry: CellEntry) -> list[tuple[Cell, list[int], float, Any]]:
        """Explore from a cell by taking random actions.

        Returns list of (cell, trajectory, score, env_state) for newly discovered cells.
        """
        # Restore the environment to this cell's state
        if entry.env_state is not None:
            obs = self._restore_env_state(entry.env_state)
        else:
            # Fallback: replay the action trajectory from reset
            obs_np, _ = self.env.reset()
            obs = obs_np
            for a in entry.trajectory:
                obs, _, terminated, truncated, _ = self.env.step(a)
                if terminated or truncated:
                    return []

        current_trajectory = list(entry.trajectory)
        current_score = entry.score
        discoveries = []
        prev_action = None

        for step in range(self.explore_steps):
            # Sticky random actions (repeat previous with high probability)
            if prev_action is not None and np.random.random() < self.action_repeat_prob:
                action = prev_action
            else:
                action = np.random.randint(0, self.n_actions)

            obs, reward, terminated, truncated, info = self.env.step(action)
            self._global_step += 1

            current_trajectory.append(action)
            current_score += reward

            if terminated or truncated:
                self._episodes_completed += 1
                if current_score > self._best_score:
                    self._best_score = current_score
                break

            # Check for new cell
            cell = self._get_cell(obs)
            env_state = self._get_env_state()
            discoveries.append((cell, list(current_trajectory), current_score, env_state))

            prev_action = action

        return discoveries

    def run(self, verbose: bool = True) -> dict:
        """Run Go-Explore Phase 1 exploration loop.

        Returns:
            Dict with exploration statistics and best trajectory.
        """
        # Initialize archive with start state
        obs_np, _ = self.env.reset()
        start_cell = self._get_cell(obs_np)
        start_state = self._get_env_state()
        self.archive.add_cell(start_cell, trajectory=[], score=0.0, env_state=start_state)

        pbar = tqdm(total=self.total_steps, desc="Go-Explore Phase 1", disable=not verbose)

        while self._global_step < self.total_steps:
            # 1. Select a cell from the archive
            entry = self.archive.select_cell()

            # 2. Reset the environment for this exploration round
            self.env.reset()

            # 3. Explore from the selected cell
            discoveries = self._explore_from(entry)

            # 4. Update the archive with discoveries
            for cell, trajectory, score, env_state in discoveries:
                is_new = self.archive.add_cell(
                    cell, trajectory, score, env_state, self._global_step
                )
                if is_new:
                    self.archive.notify_new_discovery(entry.cell)

            pbar.update(min(self.explore_steps, self.total_steps - pbar.n))

            # Periodic logging
            if verbose and self._global_step % 50_000 < self.explore_steps:
                stats = self.archive.stats()
                pbar.set_postfix(
                    cells=stats["archive_size"],
                    best=f"{stats['best_score']:.2f}",
                )

        pbar.close()

        best_traj, best_score = self.archive.get_best_trajectory()
        return {
            "total_steps": self._global_step,
            "episodes_completed": self._episodes_completed,
            "best_score": best_score,
            "best_trajectory_length": len(best_traj),
            "best_trajectory": best_traj,
            **self.archive.stats(),
        }
