"""Proximal Policy Optimization (Schulman et al., 2017).

Custom PPO implementation following the thesis (§3.1.2), extended with
intrinsic reward support for exploration methods.

L^CLIP(θ) = E_t[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]

The total reward at each step is: r_t = r_ext + beta * r_int
where r_int comes from the pluggable exploration method.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rl_exploration_lab.networks.policy import ActorCritic


class PPOTrainer:
    """PPO trainer with exploration method integration.

    Args:
        policy: ActorCritic network.
        lr: Learning rate.
        clip_range: PPO clip range epsilon.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        n_epochs: Number of PPO epochs per update.
        device: Torch device.
    """

    def __init__(
        self,
        policy: ActorCritic,
        lr: float = 2.5e-4,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.device = device

        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    def update(self, batches: list[dict[str, torch.Tensor]]) -> dict[str, float]:
        """Run PPO update for n_epochs over the given mini-batches.

        Args:
            batches: List of mini-batch dicts from RolloutBuffer.get_batches().

        Returns:
            Dict of training metrics for logging.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            for batch in batches:
                obs = batch["obs"]
                actions = batch["actions"]
                old_log_probs = batch["log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(
                    obs, actions
                )

                # Policy loss (clipped surrogate objective)
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()

                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
        }
