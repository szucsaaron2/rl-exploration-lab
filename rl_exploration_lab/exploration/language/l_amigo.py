"""L-AMIGo: Language-conditioned AMIGo (Li et al., 2022).

Extends AMIGo by using language descriptions as goals instead of (x,y) positions.
The teacher generates language goals, and the student is rewarded for reaching
states whose oracle description matches the goal.

From the thesis (§2.2.10): "L-AMIGo extends AMIGo by using language to encode goals.
The student is rewarded if it reaches a state with the language description t,
i.e., if t = L(s_t)."

The teacher has a policy + grounding network:
- Policy: proposes language goals given current state
- Grounding: estimates likelihood the student can achieve the goal
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rl_exploration_lab.envs.language_oracle import LanguageOracle
from rl_exploration_lab.exploration.base import BaseExploration
from rl_exploration_lab.networks.encoders import CLIPEncoder


class LanguageGoalTeacher(nn.Module):
    """Teacher that proposes language goals from a fixed vocabulary.

    Instead of generating free-form text, the teacher selects from a
    vocabulary of achievable goal descriptions relevant to MiniGrid.
    """

    GOAL_VOCABULARY = [
        "reach the goal",
        "find a key",
        "open a door",
        "pick up the ball",
        "explore a new room",
        "go to the right",
        "go to the left",
        "go forward",
        "find a red object",
        "find a blue object",
        "find a green object",
        "find a yellow object",
        "reach an open door",
        "reach a closed door",
        "navigate through corridor",
        "explore empty space",
    ]

    def __init__(self, obs_dim: int = 147, hidden_dim: int = 128, embed_dim: int = 512):
        super().__init__()
        n_goals = len(self.GOAL_VOCABULARY)

        # Policy: obs → goal distribution
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_goals),
        )

        # Grounding: obs + goal_emb → feasibility score
        self.grounding = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def sample_goal(self, obs: torch.Tensor) -> tuple[int, str, torch.Tensor]:
        """Sample a goal from the vocabulary.

        Returns (goal_idx, goal_text, log_prob).
        """
        logits = self.policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        goal_idx = dist.sample()
        log_prob = dist.log_prob(goal_idx)
        goal_text = self.GOAL_VOCABULARY[goal_idx.item()]
        return goal_idx.item(), goal_text, log_prob


class LAMIGo(BaseExploration):
    """Language-conditioned AMIGo.

    Teacher proposes language goals; student gets intrinsic reward for
    reaching states matching the goal description.

    Args:
        obs_dim: Raw observation dimension.
        clip_model: CLIP model name.
        hidden_dim: Teacher network hidden size.
        teacher_lr: Teacher learning rate.
        max_goal_steps: Max steps per goal.
        oracle_verbose: Whether to use verbose oracle descriptions.
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        clip_model: str = "ViT-B/32",
        hidden_dim: int = 128,
        teacher_lr: float = 1e-3,
        max_goal_steps: int = 50,
        oracle_verbose: bool = False,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.max_goal_steps = max_goal_steps

        self.clip = CLIPEncoder(model_name=clip_model, device=device)
        self.oracle = LanguageOracle(verbose=oracle_verbose)

        self.teacher = LanguageGoalTeacher(obs_dim, hidden_dim, self.clip.embed_dim).to(device)
        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=teacher_lr)

        # Pre-encode goal vocabulary
        with torch.no_grad():
            self._goal_embeddings = self.clip.encode_text(
                LanguageGoalTeacher.GOAL_VOCABULARY
            )

        # Goal tracking
        self._current_goal_text: str | None = None
        self._current_goal_embedding: torch.Tensor | None = None
        self._current_goal_log_prob: torch.Tensor | None = None
        self._steps_since_goal = 0
        self._goals_proposed = 0
        self._goals_reached = 0

    def _check_goal_match(self, state_desc: str, goal_text: str) -> bool:
        """Check if the state description matches the goal (semantic matching).

        Uses keyword overlap as a simple matching heuristic.
        """
        goal_keywords = set(goal_text.lower().split())
        state_keywords = set(state_desc.lower().split())
        # Match if key goal words appear in the state description
        important_words = goal_keywords - {"the", "a", "an", "to", "go", "find", "reach"}
        if not important_words:
            return False
        overlap = important_words & state_keywords
        return len(overlap) >= len(important_words) * 0.5

    def compute_intrinsic_reward(
        self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            self._steps_since_goal += 1

            # Propose new goal if needed
            if self._current_goal_text is None or self._steps_since_goal >= self.max_goal_steps:
                if self._current_goal_log_prob is not None:
                    self._update_teacher(goal_reached=False)
                with torch.no_grad():
                    _, goal_text, log_prob = self.teacher.sample_goal(obs[i].unsqueeze(0))
                self._current_goal_text = goal_text
                self._current_goal_log_prob = log_prob
                self._steps_since_goal = 0
                self._goals_proposed += 1

            # Check if current state matches goal
            state_desc = self.oracle.describe_observation(next_obs[i].cpu().numpy())
            if self._check_goal_match(state_desc, self._current_goal_text):
                # Goal reached — reward based on speed
                rewards[i] = max(0.0, 1.0 - 0.9 * self._steps_since_goal / self.max_goal_steps)
                self._update_teacher(goal_reached=True)
                self._current_goal_text = None  # Will propose new goal next step

        return rewards

    def _update_teacher(self, goal_reached: bool):
        """REINFORCE update for the teacher."""
        if self._current_goal_log_prob is None:
            return

        if goal_reached:
            teacher_reward = 1.0
            self._goals_reached += 1
        else:
            teacher_reward = -0.5

        loss = -self._current_goal_log_prob * teacher_reward
        self.teacher_optimizer.zero_grad()
        loss.backward()
        self.teacher_optimizer.step()

        self._current_goal_log_prob = None

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        return {
            "goals_proposed": self._goals_proposed,
            "goals_reached": self._goals_reached,
            "goal_reach_rate": self._goals_reached / max(self._goals_proposed, 1),
        }

    def state_dict(self) -> dict:
        return {
            "teacher": self.teacher.state_dict(),
            "optimizer": self.teacher_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.teacher.load_state_dict(state["teacher"])
        self.teacher_optimizer.load_state_dict(state["optimizer"])
