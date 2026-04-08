"""Unit tests for exploration methods and core components."""

import pytest
import torch
import numpy as np


class TestEpsilonGreedy:
    def test_returns_zero_reward(self):
        from rl_exploration_lab.exploration.epsilon_greedy import EpsilonGreedy

        eg = EpsilonGreedy()
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))

        reward = eg.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)
        assert (reward == 0).all()

    def test_update_returns_empty(self):
        from rl_exploration_lab.exploration.epsilon_greedy import EpsilonGreedy

        eg = EpsilonGreedy()
        metrics = eg.update({"obs": torch.randn(8, 147)})
        assert metrics == {}


class TestCountBased:
    def test_reward_decreases_with_visits(self):
        from rl_exploration_lab.exploration.count_based import CountBased

        cb = CountBased(beta=1.0)
        obs = torch.zeros(1, 147)
        next_obs = torch.ones(1, 147) * 0.5  # same state each time
        action = torch.zeros(1, dtype=torch.long)

        r1 = cb.compute_intrinsic_reward(obs, next_obs, action).item()
        r2 = cb.compute_intrinsic_reward(obs, next_obs, action).item()
        r3 = cb.compute_intrinsic_reward(obs, next_obs, action).item()

        # Reward should decrease as visit count increases
        assert r1 > r2 > r3

    def test_novel_states_get_higher_reward(self):
        from rl_exploration_lab.exploration.count_based import CountBased

        cb = CountBased(beta=1.0)
        obs = torch.zeros(1, 147)
        action = torch.zeros(1, dtype=torch.long)

        # Visit state A multiple times
        state_a = torch.ones(1, 147) * 0.3
        for _ in range(10):
            cb.compute_intrinsic_reward(obs, state_a, action)

        # State B is novel
        state_b = torch.ones(1, 147) * 0.7
        r_a = cb.compute_intrinsic_reward(obs, state_a, action).item()
        r_b = cb.compute_intrinsic_reward(obs, state_b, action).item()

        assert r_b > r_a

    def test_update_reports_stats(self):
        from rl_exploration_lab.exploration.count_based import CountBased

        cb = CountBased()
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.zeros(4, dtype=torch.long)
        cb.compute_intrinsic_reward(obs, next_obs, action)

        metrics = cb.update({"obs": obs})
        assert "unique_states" in metrics
        assert "total_visits" in metrics


class TestRND:
    def test_intrinsic_reward_shape(self):
        from rl_exploration_lab.exploration.rnd import RND

        rnd = RND(obs_dim=147, output_dim=32, hidden_dim=64, n_layers=2)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))

        reward = rnd.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)
        assert (reward >= 0).all()

    def test_reward_is_clipped(self):
        from rl_exploration_lab.exploration.rnd import RND

        rnd = RND(obs_dim=147, reward_clip=1.0)
        obs = torch.randn(16, 147)
        next_obs = torch.randn(16, 147) * 100  # large values → high prediction error
        action = torch.randint(0, 7, (16,))

        reward = rnd.compute_intrinsic_reward(obs, next_obs, action)
        assert (reward <= 1.0).all()

    def test_update_reduces_loss(self):
        from rl_exploration_lab.exploration.rnd import RND

        rnd = RND(obs_dim=147, lr=0.01)
        fixed_obs = torch.randn(32, 147)

        # Compute initial loss
        loss_before = rnd.rnd.compute_loss(fixed_obs).item()

        # Update multiple times on the same data
        for _ in range(50):
            rnd.update({"next_obs": fixed_obs})

        loss_after = rnd.rnd.compute_loss(fixed_obs).item()
        assert loss_after < loss_before

    def test_state_dict_roundtrip(self):
        from rl_exploration_lab.exploration.rnd import RND

        rnd1 = RND(obs_dim=147, output_dim=32)
        state = rnd1.state_dict()

        rnd2 = RND(obs_dim=147, output_dim=32)
        rnd2.load_state_dict(state)

        # Check weights match
        for p1, p2 in zip(rnd1.rnd.predictor.parameters(), rnd2.rnd.predictor.parameters()):
            assert torch.allclose(p1, p2)


class TestActorCritic:
    def test_forward_shapes(self):
        from rl_exploration_lab.networks.policy import ActorCritic

        policy = ActorCritic(obs_dim=147, n_actions=7)
        obs = torch.randn(8, 147)

        dist, value = policy(obs)
        assert value.shape == (8, 1)
        assert dist.probs.shape == (8, 7)

    def test_get_action_and_value_sample(self):
        from rl_exploration_lab.networks.policy import ActorCritic

        policy = ActorCritic(obs_dim=147, n_actions=7)
        obs = torch.randn(4, 147)

        action, log_prob, entropy, value = policy.get_action_and_value(obs)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_get_action_and_value_with_action(self):
        from rl_exploration_lab.networks.policy import ActorCritic

        policy = ActorCritic(obs_dim=147, n_actions=7)
        obs = torch.randn(4, 147)
        given_action = torch.tensor([0, 3, 6, 2])

        action, log_prob, entropy, value = policy.get_action_and_value(obs, given_action)
        assert torch.equal(action, given_action)


class TestRNDModule:
    def test_target_is_frozen(self):
        from rl_exploration_lab.networks.predictors import RNDModule

        rnd = RNDModule(input_dim=147, output_dim=32)
        for param in rnd.target.parameters():
            assert not param.requires_grad

    def test_predictor_is_trainable(self):
        from rl_exploration_lab.networks.predictors import RNDModule

        rnd = RNDModule(input_dim=147, output_dim=32)
        for param in rnd.predictor.parameters():
            assert param.requires_grad

    def test_intrinsic_reward_positive(self):
        from rl_exploration_lab.networks.predictors import RNDModule

        rnd = RNDModule(input_dim=147, output_dim=32)
        obs = torch.randn(8, 147)
        reward = rnd.compute_intrinsic_reward(obs)
        assert (reward >= 0).all()


class TestDynamicsModel:
    def test_icm_losses_shapes(self):
        from rl_exploration_lab.networks.dynamics import DynamicsModel

        model = DynamicsModel(obs_dim=147, n_actions=7, embed_dim=32)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))

        fwd_loss, inv_loss, intrinsic_reward = model.compute_icm_losses(obs, next_obs, action)
        assert fwd_loss.shape == ()
        assert inv_loss.shape == ()
        assert intrinsic_reward.shape == (8,)


class TestRolloutBuffer:
    def test_add_and_full(self):
        from rl_exploration_lab.training.rollout import RolloutBuffer

        buf = RolloutBuffer(buffer_size=4, obs_dim=10)
        for i in range(4):
            buf.add(
                obs=torch.zeros(10),
                next_obs=torch.zeros(10),
                action=torch.tensor(0),
                ext_reward=1.0,
                int_reward=0.1,
                done=False,
                log_prob=torch.tensor(0.0),
                value=torch.tensor(0.0),
            )
        assert buf.is_full

    def test_compute_advantages(self):
        from rl_exploration_lab.training.rollout import RolloutBuffer

        buf = RolloutBuffer(buffer_size=4, obs_dim=10, intrinsic_coef=0.0)
        for i in range(4):
            buf.add(
                obs=torch.zeros(10),
                next_obs=torch.zeros(10),
                action=torch.tensor(0),
                ext_reward=1.0,
                int_reward=0.0,
                done=False,
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.5),
            )
        buf.compute_advantages(last_value=torch.tensor(0.5), last_done=False)
        # Advantages should be computed (non-zero somewhere)
        assert buf.advantages.shape == (4,)
        assert buf.returns.shape == (4,)

    def test_get_batches(self):
        from rl_exploration_lab.training.rollout import RolloutBuffer

        buf = RolloutBuffer(buffer_size=8, obs_dim=10)
        for i in range(8):
            buf.add(
                obs=torch.randn(10),
                next_obs=torch.randn(10),
                action=torch.tensor(i % 7),
                ext_reward=0.0,
                int_reward=0.0,
                done=False,
                log_prob=torch.tensor(-1.0),
                value=torch.tensor(0.0),
            )
        buf.compute_advantages(torch.tensor(0.0), False)
        batches = buf.get_batches(batch_size=4)
        assert len(batches) == 2
        assert batches[0]["obs"].shape == (4, 10)


class TestEnvRegistry:
    def test_list_envs(self):
        from rl_exploration_lab.envs.env_registry import list_envs

        envs = list_envs()
        assert "KeyCorridorS3R2" in envs
        assert "Empty-8x8" in envs
        assert len(envs) == 5

    def test_make_env(self):
        from rl_exploration_lab.envs.env_registry import make_env

        env = make_env("Empty-8x8", seed=42)
        obs, info = env.reset()
        assert "image" in obs
        env.close()


class TestMiniGridWrapper:
    def test_wrapped_obs_shape(self):
        from rl_exploration_lab.envs.minigrid_wrapper import make_wrapped_env

        env = make_wrapped_env("Empty-8x8", seed=0)
        obs, info = env.reset()
        assert obs.shape == (147,)
        assert obs.dtype == np.float32
        env.close()

    def test_step_returns_correct_shapes(self):
        from rl_exploration_lab.envs.minigrid_wrapper import make_wrapped_env

        env = make_wrapped_env("Empty-8x8", seed=0)
        obs, _ = env.reset()
        next_obs, reward, terminated, truncated, info = env.step(2)  # forward
        assert next_obs.shape == (147,)
        assert isinstance(reward, (int, float))
        env.close()

    def test_agent_state(self):
        from rl_exploration_lab.envs.minigrid_wrapper import make_wrapped_env

        env = make_wrapped_env("Empty-8x8", seed=0)
        env.reset()
        state = env.get_agent_state()
        assert "pos" in state
        assert "dir" in state
        assert len(state["pos"]) == 2
        env.close()
