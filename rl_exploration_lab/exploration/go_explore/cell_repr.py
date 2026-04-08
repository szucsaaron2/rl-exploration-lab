"""Cell representations for Go-Explore (Ecoffet et al., 2019/2021).

A cell representation maps high-dimensional states to a lower-dimensional
space. Similar states should map to the same cell, while meaningfully
different states should map to different cells.

Provides:
- DownsampledImageCell: domain-agnostic, downsamples the observation
- MiniGridDomainCell: uses agent position, direction, and inventory
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cell:
    """A hashable cell representation for the Go-Explore archive.

    Attributes:
        key: Hashable representation of this cell (e.g., tuple of ints).
        metadata: Optional non-hashable metadata (not used for equality/hashing).
    """
    key: tuple

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Cell):
            return NotImplemented
        return self.key == other.key


class CellRepresentation:
    """Base class for cell representation strategies."""

    def obs_to_cell(self, obs: np.ndarray, env_state: dict | None = None) -> Cell:
        """Map an observation (and optional env state) to a Cell."""
        raise NotImplementedError


class DownsampledImageCell(CellRepresentation):
    """Domain-agnostic cell representation via observation downsampling.

    Following Go-Explore's approach: discretize the observation to reduce
    the state space while preserving meaningful distinctions.

    For MiniGrid's 7x7x3 = 147-dim observations, we discretize the float
    values into a small number of bins.

    Args:
        n_bins: Number of intensity bins for discretization.
    """

    def __init__(self, n_bins: int = 8):
        self.n_bins = n_bins

    def obs_to_cell(self, obs: np.ndarray, env_state: dict | None = None) -> Cell:
        """Discretize the observation into bins and create a hashable cell."""
        # obs is already in [0, 1] from the MiniGrid wrapper
        discretized = np.floor(obs * self.n_bins).astype(np.int8)
        discretized = np.clip(discretized, 0, self.n_bins - 1)
        return Cell(key=tuple(discretized.tolist()))


class MiniGridDomainCell(CellRepresentation):
    """MiniGrid-specific cell representation using domain knowledge.

    Uses the agent's position, direction, and carried object to define cells.
    This is much more compact than the full observation and captures the
    semantically meaningful state differences.

    Requires env_state dict from MiniGridWrapper.get_agent_state().
    """

    def obs_to_cell(self, obs: np.ndarray, env_state: dict | None = None) -> Cell:
        """Create cell from agent state (position, direction, carrying).

        Args:
            obs: Flat observation (used as fallback if env_state not available).
            env_state: Dict with 'pos', 'dir', 'carrying' from get_agent_state().
        """
        if env_state is not None:
            key = (
                env_state["pos"],      # (x, y) tuple
                env_state["dir"],      # 0-3 direction
                env_state.get("carrying"),  # carried object string or None
            )
            return Cell(key=key)
        else:
            # Fallback: use downsampled observation
            return DownsampledImageCell(n_bins=8).obs_to_cell(obs)
