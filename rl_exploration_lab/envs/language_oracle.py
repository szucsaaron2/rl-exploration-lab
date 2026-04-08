"""Language oracle for MiniGrid environments.

Provides ground-truth natural language descriptions of MiniGrid states.
This is the "language oracle" L(s) referenced in the thesis and in
Li et al. (2022) "Improving Intrinsic Exploration via Language Abstractions".

The oracle extracts visible objects from the agent's 7x7 partial view and
generates a structured text description. This enables:
- L-AMIGo: language-conditioned goal generation
- L-NovelD: language-based novelty measurement
- SHELM + Oracle: the thesis's proposed improvement (Chapter 6)

MiniGrid object encoding: each tile is (object_type, color, state)
Object types: 0=unseen, 1=empty, 2=wall, 3=floor, 4=door,
5=key, 6=ball, 7=box, 8=goal, 9=lava, 10=agent
Colors: 0=red, 1=green, 2=blue, 3=purple, 4=yellow, 5=grey
Door states: 0=open, 1=closed, 2=locked
"""

from __future__ import annotations

import numpy as np

# MiniGrid object type names
OBJECT_NAMES = {
    0: "unseen",
    1: "empty",
    2: "wall",
    3: "floor",
    4: "door",
    5: "key",
    6: "ball",
    7: "box",
    8: "goal",
    9: "lava",
    10: "agent",
}

COLOR_NAMES = {
    0: "red",
    1: "green",
    2: "blue",
    3: "purple",
    4: "yellow",
    5: "grey",
}

DOOR_STATES = {
    0: "open",
    1: "closed",
    2: "locked",
}

DIRECTION_NAMES = {
    0: "right",
    1: "down",
    2: "left",
    3: "up",
}


class LanguageOracle:
    """Generates natural language descriptions of MiniGrid states.

    Takes the agent's 7x7x3 partial observation grid and produces a
    structured text description of what the agent can see.

    Args:
        include_positions: Whether to include relative positions in descriptions.
        include_direction: Whether to include the agent's facing direction.
        verbose: If True, generates detailed descriptions; if False, concise tokens.
    """

    def __init__(
        self,
        include_positions: bool = True,
        include_direction: bool = True,
        verbose: bool = True,
    ):
        self.include_positions = include_positions
        self.include_direction = include_direction
        self.verbose = verbose

    def describe_observation(
        self,
        obs: np.ndarray,
        agent_dir: int | None = None,
        carrying: str | None = None,
    ) -> str:
        """Generate a text description of a MiniGrid observation.

        Args:
            obs: Flat observation array of shape (147,) in [0, 1], or
                 raw grid of shape (7, 7, 3) with integer values.
            agent_dir: Agent's facing direction (0-3). Optional.
            carrying: What the agent is carrying (string or None).

        Returns:
            Natural language description of the observation.
        """
        # Handle flat normalized obs from MiniGridWrapper
        if obs.shape == (147,):
            grid = (obs * 255).reshape(7, 7, 3).astype(np.int32)
        elif obs.shape == (7, 7, 3):
            grid = obs.astype(np.int32)
        else:
            return "Unknown observation format."

        if self.verbose:
            return self._describe_verbose(grid, agent_dir, carrying)
        else:
            return self._describe_concise(grid, agent_dir, carrying)

    def _describe_verbose(
        self, grid: np.ndarray, agent_dir: int | None, carrying: str | None
    ) -> str:
        """Generate a detailed natural language description."""
        parts = []

        # Agent status
        if agent_dir is not None:
            dir_name = DIRECTION_NAMES.get(agent_dir, "unknown")
            parts.append(f"The agent is facing {dir_name}.")

        if carrying:
            parts.append(f"The agent is carrying a {carrying}.")

        # Scan visible objects (exclude walls, empty, unseen, floor)
        interesting_objects = []
        for y in range(7):
            for x in range(7):
                obj_type = grid[y, x, 0]
                color = grid[y, x, 1]
                state = grid[y, x, 2]

                if obj_type in (0, 1, 2, 3):  # unseen, empty, wall, floor
                    continue

                obj_name = OBJECT_NAMES.get(obj_type, f"object_{obj_type}")
                color_name = COLOR_NAMES.get(color, f"color_{color}")

                # Relative position from agent (agent is at 3, 6 in the 7x7 view)
                rel_x = x - 3
                rel_y = 6 - y  # y increases downward in grid, upward in description

                desc = f"{color_name} {obj_name}"
                if obj_type == 4:  # door
                    door_state = DOOR_STATES.get(state, "unknown")
                    desc = f"{door_state} {desc}"

                if self.include_positions:
                    pos_desc = self._position_words(rel_x, rel_y)
                    desc = f"{desc} {pos_desc}"

                interesting_objects.append(desc)

        if interesting_objects:
            objects_text = "Visible: " + ", ".join(interesting_objects) + "."
            parts.append(objects_text)
        else:
            parts.append("No interesting objects visible.")

        return " ".join(parts)

    def _describe_concise(
        self, grid: np.ndarray, agent_dir: int | None, carrying: str | None
    ) -> str:
        """Generate a concise token-style description."""
        tokens = []

        if agent_dir is not None:
            tokens.append(DIRECTION_NAMES.get(agent_dir, "?"))

        if carrying:
            tokens.append(f"carry:{carrying}")

        for y in range(7):
            for x in range(7):
                obj_type = grid[y, x, 0]
                color = grid[y, x, 1]
                state = grid[y, x, 2]

                if obj_type in (0, 1, 2, 3):
                    continue

                obj_name = OBJECT_NAMES.get(obj_type, f"obj{obj_type}")
                color_name = COLOR_NAMES.get(color, f"c{color}")

                token = f"{color_name}_{obj_name}"
                if obj_type == 4:
                    door_state = DOOR_STATES.get(state, "?")
                    token = f"{door_state}_{token}"

                tokens.append(token)

        return " ".join(tokens) if tokens else "empty"

    def _position_words(self, rel_x: int, rel_y: int) -> str:
        """Convert relative position to natural language."""
        parts = []
        if rel_y > 0:
            parts.append(f"{rel_y} ahead")
        elif rel_y < 0:
            parts.append(f"{-rel_y} behind")

        if rel_x > 0:
            parts.append(f"{rel_x} right")
        elif rel_x < 0:
            parts.append(f"{-rel_x} left")

        if not parts:
            return "directly in front"

        return ", ".join(parts)

    def describe_full_state(
        self,
        obs: np.ndarray,
        agent_pos: tuple[int, int] | None = None,
        agent_dir: int | None = None,
        carrying: str | None = None,
    ) -> str:
        """Generate a description including absolute position (if available).

        This provides richer descriptions for methods that benefit from
        global state information (like SHELM + Oracle).
        """
        base = self.describe_observation(obs, agent_dir, carrying)
        if agent_pos is not None:
            base = f"Agent at position ({agent_pos[0]}, {agent_pos[1]}). " + base
        return base
