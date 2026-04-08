"""AMIGo: Adversarially Motivated Intrinsic Goals (Campero et al., 2020).

A teacher-student architecture where:
- The teacher generates goals for the student (e.g., "go to position (x,y)")
- The student learns to reach those goals
- The teacher is rewarded for proposing goals that are challenging but achievable

Teacher reward:
    r_T = +α  if student takes ≥ t* steps to reach goal (challenging)
    r_T = -β  if student fails or reaches too quickly (too easy/impossible)

Student intrinsic reward:
    r_i = 1 - 0.9 * t / t_max  (faster completion → higher reward)

Adapted for MiniGrid: goals are (x, y) positions on the grid.
The teacher learns which positions are good exploration targets.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rl_exploration_lab.exploration.base import BaseExploration


class GoalGenerator(nn.Module):
    """Teacher network that generates (x, y) goal positions.

    Takes the current grid observation and outputs a distribution over
    possible goal positions.

    Args:
        obs_dim: Observation dimension.
        grid_size: Maximum grid dimension (goals are positions in [0, grid_size)^2).
        hidden_dim: Hidden layer size.
    """

    def __init__(self, obs_dim: int = 147, grid_size: int = 20, hidden_dim: int = 128):
        super().__init__()
        self.grid_size = grid_size
        self.n_goals = grid_size * grid_size

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_goals),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Output logits over goal positions."""
        return self.net(obs)

    def sample_goal(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a goal position and return (goal_idx, log_prob)."""
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        goal_idx = dist.sample()
        log_prob = dist.log_prob(goal_idx)
        return goal_idx, log_prob

    def goal_idx_to_pos(self, goal_idx: int) -> tuple[int, int]:
        """Convert a flat goal index to (x, y) position."""
        x = goal_idx % self.grid_size
        y = goal_idx // self.grid_size
        return (x, y)


class AMIGo(BaseExploration):
    """AMIGo: Adversarially Motivated Intrinsic Goals.

    The teacher proposes goal positions. The student gets intrinsic reward
    for reaching them efficiently. The teacher learns which goals produce
    good learning progress.

    Args:
        obs_dim: Observation dimension.
        grid_size: Maximum grid dimension for goal positions.
        hidden_dim: Hidden layer size.
        teacher_lr: Teacher network learning rate.
        teacher_alpha: Reward for teacher when goal is challenging but reached.
        teacher_beta: Penalty when goal is too easy or not reached.
        max_goal_steps: Maximum steps allowed to reach a goal (t_max).
        challenge_threshold: Minimum steps for a goal to be "challenging" (t*).
        device: Torch device.
    """

    def __init__(
        self,
        obs_dim: int = 147,
        grid_size: int = 20,
        hidden_dim: int = 128,
        teacher_lr: float = 1e-3,
        teacher_alpha: float = 1.0,
        teacher_beta: float = 0.5,
        max_goal_steps: int = 50,
        challenge_threshold: int = 10,
        device: str = "cpu",
    ):
        super().__init__(device=device)
        self.grid_size = grid_size
        self.teacher_alpha = teacher_alpha
        self.teacher_beta = teacher_beta
        self.max_goal_steps = max_goal_steps
        self.challenge_threshold = challenge_threshold

        self.teacher = GoalGenerator(obs_dim, grid_size, hidden_dim).to(device)
        self.teacher_optimizer = optim.Adam(self.teacher.parameters(), lr=teacher_lr)

        # Current goal state
        self._current_goal: tuple[int, int] | None = None
        self._current_goal_log_prob: torch.Tensor | None = None
        self._steps_since_goal = 0
        self._goal_reached = False

        # Metrics
        self._goals_proposed = 0
        self._goals_reached = 0
        self._teacher_loss_accum = 0.0
        self._teacher_updates = 0

    def _set_new_goal(self, obs: torch.Tensor):
        """Have the teacher propose a new goal."""
        goal_idx, log_prob = self.teacher.sample_goal(obs.unsqueeze(0))
        self._current_goal = self.teacher.goal_idx_to_pos(goal_idx.item())
        self._current_goal_log_prob = log_prob
        self._steps_since_goal = 0
        self._goal_reached = False
        self._goals_proposed += 1

    def _check_goal_reached(self, agent_pos: tuple[int, int]) -> bool:
        """Check if the agent has reached the current goal position."""
        if self._current_goal is None:
            return False
        return agent_pos == self._current_goal

    def compute_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AMIGo student intrinsic reward.

        Since we can't easily extract agent position from flat obs in this
        interface, we use a proxy: reward based on observation change magnitude
        toward the goal direction. For full AMIGo, integrate with env.get_agent_state().

        For the general case, we provide a simplified version:
        - Propose a goal (encoded as an obs embedding)
        - Reward the student for reaching novel states (as a proxy for reaching goals)
        - Periodically update the teacher based on student progress

        r_i = 1 - 0.9 * t / t_max  (when goal is reached)
        r_i = 0                      (otherwise)
        """
        batch_size = obs.shape[0]
        rewards = torch.zeros(batch_size, device=self.device)

        # Simplified: use observation change as proxy for progress
        # In a full implementation, you'd check actual agent position vs goal
        with torch.no_grad():
            obs_change = (next_obs - obs).pow(2).sum(dim=-1).sqrt()

            for i in range(batch_size):
                self._steps_since_goal += 1

                # Propose new goal if needed
                if self._current_goal is None or self._steps_since_goal >= self.max_goal_steps:
                    self._update_teacher_reward(goal_reached=False)
                    self._set_new_goal(obs[i])

                # Proxy: high observation change → progress toward goal
                if obs_change[i] > 0.1:  # significant state change
                    rewards[i] = 1.0 - 0.9 * self._steps_since_goal / self.max_goal_steps
                    rewards[i] = max(rewards[i], 0.0)

        return rewards

    def _update_teacher_reward(self, goal_reached: bool):
        """Update the teacher based on whether the student reached the goal."""
        if self._current_goal_log_prob is None:
            return

        steps = self._steps_since_goal

        if goal_reached and steps >= self.challenge_threshold:
            teacher_reward = self.teacher_alpha  # Good: challenging but reached
        else:
            teacher_reward = -self.teacher_beta  # Bad: too easy, impossible, or timed out

        # REINFORCE update for the teacher
        teacher_loss = -self._current_goal_log_prob * teacher_reward

        self.teacher_optimizer.zero_grad()
        teacher_loss.backward()
        self.teacher_optimizer.step()

        self._teacher_loss_accum += teacher_loss.item()
        self._teacher_updates += 1
        if goal_reached:
            self._goals_reached += 1

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Report AMIGo metrics (teacher updates happen inline)."""
        avg_teacher_loss = (
            self._teacher_loss_accum / max(self._teacher_updates, 1)
        )

        metrics = {
            "goals_proposed": self._goals_proposed,
            "goals_reached": self._goals_reached,
            "goal_reach_rate": self._goals_reached / max(self._goals_proposed, 1),
            "teacher_loss": avg_teacher_loss,
        }

        # Reset accumulators
        self._teacher_loss_accum = 0.0
        self._teacher_updates = 0

        return metrics

    def get_exploration_loss(self) -> float | None:
        return None

    def state_dict(self) -> dict:
        return {
            "teacher": self.teacher.state_dict(),
            "teacher_optimizer": self.teacher_optimizer.state_dict(),
            "goals_proposed": self._goals_proposed,
            "goals_reached": self._goals_reached,
        }

    def load_state_dict(self, state: dict) -> None:
        self.teacher.load_state_dict(state["teacher"])
        self.teacher_optimizer.load_state_dict(state["teacher_optimizer"])
        self._goals_proposed = state.get("goals_proposed", 0)
        self._goals_reached = state.get("goals_reached", 0)
