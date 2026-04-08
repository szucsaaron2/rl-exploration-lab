"""Go-Explore archive: stores discovered cells, trajectories, and selection heuristics.

The archive is the core data structure of Go-Explore Phase 1. It stores:
- All discovered cells (interestingly different states)
- The trajectory (sequence of actions) that leads to each cell
- The simulator state at each cell (for fast restoration)
- Selection weights for prioritizing which cell to explore from next

Selection heuristic: prefer cells that are less visited, were recently
discovered, or have recently led to discovering new cells.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rl_exploration_lab.exploration.go_explore.cell_repr import Cell


@dataclass
class CellEntry:
    """An entry in the Go-Explore archive for a single cell.

    Attributes:
        cell: The cell representation.
        trajectory: Sequence of actions from the start state to this cell.
        score: Cumulative extrinsic reward along the trajectory.
        env_state: Saved simulator state for fast restoration (if available).
        times_chosen: How many times this cell has been selected for exploration.
        times_chosen_since_new: Times chosen since last leading to a new discovery.
        times_visited: Total number of times this cell has been visited.
        discovered_at: Global step when this cell was first discovered.
    """
    cell: Cell
    trajectory: list[int] = field(default_factory=list)
    score: float = 0.0
    env_state: Any = None
    times_chosen: int = 0
    times_chosen_since_new: int = 0
    times_visited: int = 1
    discovered_at: int = 0


class Archive:
    """Go-Explore cell archive with weighted selection.

    Stores discovered cells and provides weighted random selection for
    choosing which cell to explore from next.

    Args:
        weight_visits: Weight for the visit-count component (prefer less visited).
        weight_new_discovery: Weight for the discovery-recency component.
        weight_trajectory_length: Preference for shorter trajectories.
    """

    def __init__(
        self,
        weight_visits: float = 0.1,
        weight_new_discovery: float = 1.0,
        weight_trajectory_length: float = 0.3,
    ):
        self.entries: dict[Cell, CellEntry] = {}
        self.weight_visits = weight_visits
        self.weight_new_discovery = weight_new_discovery
        self.weight_trajectory_length = weight_trajectory_length

    @property
    def size(self) -> int:
        return len(self.entries)

    def add_cell(
        self,
        cell: Cell,
        trajectory: list[int],
        score: float = 0.0,
        env_state: Any = None,
        global_step: int = 0,
    ) -> bool:
        """Add a new cell or update an existing one if the trajectory is better.

        A trajectory is "better" if it has a higher score or is shorter
        with the same score.

        Args:
            cell: Cell representation.
            trajectory: Action sequence from start to this cell.
            score: Cumulative reward of the trajectory.
            env_state: Simulator state at this cell.
            global_step: Current global step counter.

        Returns:
            True if the cell was new or updated, False if skipped.
        """
        if cell not in self.entries:
            # New cell
            self.entries[cell] = CellEntry(
                cell=cell,
                trajectory=list(trajectory),
                score=score,
                env_state=env_state,
                discovered_at=global_step,
            )
            return True
        else:
            existing = self.entries[cell]
            # Update if better trajectory (higher score, or same score + shorter)
            is_better = (
                score > existing.score
                or (score == existing.score and len(trajectory) < len(existing.trajectory))
            )
            if is_better:
                existing.trajectory = list(trajectory)
                existing.score = score
                existing.env_state = env_state
                # Reset selection counts (new trajectory might be better stepping stone)
                existing.times_chosen = 0
                existing.times_chosen_since_new = 0
                return True

            existing.times_visited += 1
            return False

    def select_cell(self) -> CellEntry:
        """Select a cell to explore from using weighted random sampling.

        Weight formula (higher = more likely to be selected):
            w = 1 / (visit_count^w_v * (1 + times_chosen_since_new)^w_n
                      * trajectory_length^w_t)

        All cells have non-zero weight so no cell is permanently excluded.

        Returns:
            Selected CellEntry.
        """
        if not self.entries:
            raise RuntimeError("Archive is empty — cannot select a cell.")

        entries = list(self.entries.values())

        weights = np.array([
            1.0 / (
                max(e.times_visited, 1) ** self.weight_visits
                * (1 + e.times_chosen_since_new) ** self.weight_new_discovery
                * max(len(e.trajectory), 1) ** self.weight_trajectory_length
            )
            for e in entries
        ])

        # Normalize to probabilities
        total = weights.sum()
        if total <= 0:
            probs = np.ones(len(entries)) / len(entries)
        else:
            probs = weights / total

        idx = np.random.choice(len(entries), p=probs)
        selected = entries[idx]
        selected.times_chosen += 1
        selected.times_chosen_since_new += 1
        return selected

    def notify_new_discovery(self, from_cell: Cell):
        """Reset the 'times chosen since new' counter for a cell that led to a discovery."""
        if from_cell in self.entries:
            self.entries[from_cell].times_chosen_since_new = 0

    def get_best_trajectory(self) -> tuple[list[int], float]:
        """Return the trajectory with the highest score in the archive.

        Returns:
            (trajectory, score) tuple.
        """
        if not self.entries:
            return [], 0.0

        best = max(self.entries.values(), key=lambda e: e.score)
        return best.trajectory, best.score

    def stats(self) -> dict[str, float]:
        """Return archive statistics."""
        if not self.entries:
            return {"archive_size": 0, "best_score": 0.0, "mean_trajectory_len": 0.0}

        entries = list(self.entries.values())
        return {
            "archive_size": len(entries),
            "best_score": max(e.score for e in entries),
            "mean_trajectory_len": np.mean([len(e.trajectory) for e in entries]),
            "max_trajectory_len": max(len(e.trajectory) for e in entries),
            "total_visits": sum(e.times_visited for e in entries),
        }
