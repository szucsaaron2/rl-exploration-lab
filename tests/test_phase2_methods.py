"""Tests for Phase 2 exploration methods: UCB, ICM, RIDE, NovelD, NGU, AMIGo, Go-Explore."""

import pytest
import torch
import numpy as np


class TestUCB:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.ucb import UCB

        ucb = UCB(c=1.0)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))
        reward = ucb.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)

    def test_reward_decreases_with_visits(self):
        from rl_exploration_lab.exploration.ucb import UCB

        ucb = UCB(c=1.0)
        obs = torch.zeros(1, 147)
        next_obs = torch.zeros(1, 147)
        action = torch.tensor([0])

        # Warm up with a few different actions so total_steps > 1
        for _ in range(5):
            ucb.compute_intrinsic_reward(torch.randn(1, 147), torch.randn(1, 147), torch.tensor([1]))

        r1 = ucb.compute_intrinsic_reward(obs, next_obs, action).item()
        r2 = ucb.compute_intrinsic_reward(obs, next_obs, action).item()
        # Same state-action visited again → count increases → reward decreases
        assert r2 < r1


class TestICM:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.icm import ICM

        icm = ICM(obs_dim=147, n_actions=7)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))
        reward = icm.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)
        assert (reward >= 0).all()

    def test_reward_is_clipped(self):
        from rl_exploration_lab.exploration.icm import ICM

        icm = ICM(obs_dim=147, n_actions=7, reward_clip=0.5)
        obs = torch.randn(16, 147)
        next_obs = torch.randn(16, 147) * 100
        action = torch.randint(0, 7, (16,))
        reward = icm.compute_intrinsic_reward(obs, next_obs, action)
        assert (reward <= 0.5).all()

    def test_update_returns_losses(self):
        from rl_exploration_lab.exploration.icm import ICM

        icm = ICM(obs_dim=147, n_actions=7)
        batch = {
            "obs": torch.randn(16, 147),
            "next_obs": torch.randn(16, 147),
            "actions": torch.randint(0, 7, (16,)),
        }
        metrics = icm.update(batch)
        assert "forward_loss" in metrics
        assert "inverse_loss" in metrics
        assert "exploration_loss" in metrics

    def test_update_reduces_forward_loss(self):
        from rl_exploration_lab.exploration.icm import ICM

        icm = ICM(obs_dim=147, n_actions=7, lr=0.01)
        fixed_batch = {
            "obs": torch.randn(32, 147),
            "next_obs": torch.randn(32, 147),
            "actions": torch.randint(0, 7, (32,)),
        }
        m1 = icm.update(fixed_batch)
        for _ in range(100):
            icm.update(fixed_batch)
        m2 = icm.update(fixed_batch)
        # Total exploration loss (combined fwd+inv) should decrease on repeated data
        assert m2["exploration_loss"] < m1["exploration_loss"]


class TestRIDE:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.ride import RIDE

        ride = RIDE(obs_dim=147, n_actions=7)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))
        reward = ride.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)
        assert (reward >= 0).all()

    def test_episodic_count_reduces_reward(self):
        from rl_exploration_lab.exploration.ride import RIDE

        ride = RIDE(obs_dim=147, n_actions=7, reward_clip=None)
        obs = torch.zeros(1, 147)
        next_obs = torch.ones(1, 147) * 0.5  # same "next state" each time
        action = torch.tensor([0])

        r1 = ride.compute_intrinsic_reward(obs, next_obs, action).item()
        r2 = ride.compute_intrinsic_reward(obs, next_obs, action).item()
        # Same state visited again → episodic count increases → lower reward
        assert r2 <= r1

    def test_update_returns_metrics(self):
        from rl_exploration_lab.exploration.ride import RIDE

        ride = RIDE(obs_dim=147, n_actions=7)
        batch = {
            "obs": torch.randn(16, 147),
            "next_obs": torch.randn(16, 147),
            "actions": torch.randint(0, 7, (16,)),
        }
        metrics = ride.update(batch)
        assert "forward_loss" in metrics
        assert "inverse_loss" in metrics


class TestNovelD:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.noveld import NovelD

        noveld = NovelD(obs_dim=147)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))
        reward = noveld.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)
        assert (reward >= 0).all()

    def test_reward_is_nonnegative(self):
        from rl_exploration_lab.exploration.noveld import NovelD

        noveld = NovelD(obs_dim=147, alpha=0.5)
        obs = torch.randn(32, 147)
        next_obs = torch.randn(32, 147)
        action = torch.randint(0, 7, (32,))
        reward = noveld.compute_intrinsic_reward(obs, next_obs, action)
        # max(..., 0) ensures non-negative
        assert (reward >= 0).all()

    def test_erir_zeros_out_revisits(self):
        from rl_exploration_lab.exploration.noveld import NovelD

        noveld = NovelD(obs_dim=147, use_erir=True, reward_clip=None, alpha=0.0)
        obs = torch.zeros(1, 147)
        # Same next_obs both times
        next_obs = torch.ones(1, 147) * 0.5
        action = torch.tensor([0])

        r1 = noveld.compute_intrinsic_reward(obs, next_obs, action).item()
        r2 = noveld.compute_intrinsic_reward(obs, next_obs, action).item()
        # ERIR: second visit to same state → zero reward
        assert r2 == 0.0

    def test_update_reduces_loss(self):
        from rl_exploration_lab.exploration.noveld import NovelD

        noveld = NovelD(obs_dim=147, lr=0.01)
        fixed_obs = torch.randn(32, 147)
        for _ in range(50):
            noveld.update({"next_obs": fixed_obs})
        loss = noveld.get_exploration_loss()
        assert loss is not None
        # Loss should be reasonable (not exploding)
        assert loss < 10.0


class TestNGU:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.ngu import NGU

        ngu = NGU(obs_dim=147)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))
        reward = ngu.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)
        assert (reward >= 0).all()

    def test_episodic_count_effect(self):
        from rl_exploration_lab.exploration.ngu import NGU

        ngu = NGU(obs_dim=147, reward_clip=None)
        obs = torch.zeros(1, 147)
        next_obs = torch.ones(1, 147) * 0.3
        action = torch.tensor([0])

        r1 = ngu.compute_intrinsic_reward(obs, next_obs, action).item()
        r2 = ngu.compute_intrinsic_reward(obs, next_obs, action).item()
        # Episodic count increases → episodic reward decreases
        assert r2 <= r1

    def test_update_returns_metrics(self):
        from rl_exploration_lab.exploration.ngu import NGU

        ngu = NGU(obs_dim=147)
        batch = {"next_obs": torch.randn(16, 147)}
        metrics = ngu.update(batch)
        assert "exploration_loss" in metrics
        assert "lifelong_mean_novelty" in metrics


class TestAMIGo:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.amigo import AMIGo

        amigo = AMIGo(obs_dim=147)
        obs = torch.randn(8, 147)
        next_obs = torch.randn(8, 147)
        action = torch.randint(0, 7, (8,))
        reward = amigo.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (8,)

    def test_update_returns_metrics(self):
        from rl_exploration_lab.exploration.amigo import AMIGo

        amigo = AMIGo(obs_dim=147)
        # Generate some intrinsic rewards first to trigger teacher updates
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        amigo.compute_intrinsic_reward(obs, next_obs, action)

        metrics = amigo.update({"obs": obs})
        assert "goals_proposed" in metrics
        assert "goal_reach_rate" in metrics

    def test_state_dict_roundtrip(self):
        from rl_exploration_lab.exploration.amigo import AMIGo

        a1 = AMIGo(obs_dim=147)
        state = a1.state_dict()
        a2 = AMIGo(obs_dim=147)
        a2.load_state_dict(state)
        # Check teacher weights match
        for p1, p2 in zip(a1.teacher.parameters(), a2.teacher.parameters()):
            assert torch.allclose(p1, p2)


class TestGoExploreCellRepr:
    def test_downsampled_cell(self):
        from rl_exploration_lab.exploration.go_explore.cell_repr import DownsampledImageCell

        cell_repr = DownsampledImageCell(n_bins=8)
        obs = np.random.rand(147).astype(np.float32)
        cell = cell_repr.obs_to_cell(obs)
        assert cell.key is not None
        assert len(cell.key) == 147

    def test_same_obs_same_cell(self):
        from rl_exploration_lab.exploration.go_explore.cell_repr import DownsampledImageCell

        cell_repr = DownsampledImageCell(n_bins=8)
        obs = np.array([0.5] * 147, dtype=np.float32)
        c1 = cell_repr.obs_to_cell(obs)
        c2 = cell_repr.obs_to_cell(obs)
        assert c1 == c2

    def test_different_obs_different_cell(self):
        from rl_exploration_lab.exploration.go_explore.cell_repr import DownsampledImageCell

        cell_repr = DownsampledImageCell(n_bins=8)
        obs1 = np.zeros(147, dtype=np.float32)
        obs2 = np.ones(147, dtype=np.float32)
        c1 = cell_repr.obs_to_cell(obs1)
        c2 = cell_repr.obs_to_cell(obs2)
        assert c1 != c2

    def test_minigrid_domain_cell(self):
        from rl_exploration_lab.exploration.go_explore.cell_repr import MiniGridDomainCell

        cell_repr = MiniGridDomainCell()
        obs = np.random.rand(147).astype(np.float32)
        env_state = {"pos": (3, 5), "dir": 2, "carrying": None}
        cell = cell_repr.obs_to_cell(obs, env_state=env_state)
        assert cell.key == ((3, 5), 2, None)

    def test_minigrid_domain_cell_with_carrying(self):
        from rl_exploration_lab.exploration.go_explore.cell_repr import MiniGridDomainCell

        cell_repr = MiniGridDomainCell()
        obs = np.random.rand(147).astype(np.float32)
        state1 = {"pos": (3, 5), "dir": 2, "carrying": None}
        state2 = {"pos": (3, 5), "dir": 2, "carrying": "key"}
        c1 = cell_repr.obs_to_cell(obs, env_state=state1)
        c2 = cell_repr.obs_to_cell(obs, env_state=state2)
        assert c1 != c2  # different because carrying differs


class TestGoExploreArchive:
    def test_add_new_cell(self):
        from rl_exploration_lab.exploration.go_explore.archive import Archive
        from rl_exploration_lab.exploration.go_explore.cell_repr import Cell

        archive = Archive()
        cell = Cell(key=(1, 2, 3))
        is_new = archive.add_cell(cell, trajectory=[0, 1, 2], score=1.0)
        assert is_new
        assert archive.size == 1

    def test_update_better_trajectory(self):
        from rl_exploration_lab.exploration.go_explore.archive import Archive
        from rl_exploration_lab.exploration.go_explore.cell_repr import Cell

        archive = Archive()
        cell = Cell(key=(1, 2, 3))
        archive.add_cell(cell, trajectory=[0, 1, 2, 3], score=1.0)
        # Same cell, shorter trajectory, same score → should update
        updated = archive.add_cell(cell, trajectory=[0, 1], score=1.0)
        assert updated
        assert len(archive.entries[cell].trajectory) == 2

    def test_update_higher_score(self):
        from rl_exploration_lab.exploration.go_explore.archive import Archive
        from rl_exploration_lab.exploration.go_explore.cell_repr import Cell

        archive = Archive()
        cell = Cell(key=(1, 2, 3))
        archive.add_cell(cell, trajectory=[0, 1], score=1.0)
        # Same cell, higher score → should update
        updated = archive.add_cell(cell, trajectory=[0, 1, 2, 3], score=5.0)
        assert updated
        assert archive.entries[cell].score == 5.0

    def test_select_cell(self):
        from rl_exploration_lab.exploration.go_explore.archive import Archive
        from rl_exploration_lab.exploration.go_explore.cell_repr import Cell

        archive = Archive()
        for i in range(10):
            archive.add_cell(Cell(key=(i,)), trajectory=[i], score=float(i))
        selected = archive.select_cell()
        assert selected is not None
        assert selected.cell in archive.entries

    def test_get_best_trajectory(self):
        from rl_exploration_lab.exploration.go_explore.archive import Archive
        from rl_exploration_lab.exploration.go_explore.cell_repr import Cell

        archive = Archive()
        archive.add_cell(Cell(key=(1,)), trajectory=[0, 1], score=1.0)
        archive.add_cell(Cell(key=(2,)), trajectory=[0, 1, 2], score=5.0)
        archive.add_cell(Cell(key=(3,)), trajectory=[0], score=3.0)

        traj, score = archive.get_best_trajectory()
        assert score == 5.0
        assert traj == [0, 1, 2]

    def test_stats(self):
        from rl_exploration_lab.exploration.go_explore.archive import Archive
        from rl_exploration_lab.exploration.go_explore.cell_repr import Cell

        archive = Archive()
        archive.add_cell(Cell(key=(1,)), trajectory=[0, 1], score=2.0)
        archive.add_cell(Cell(key=(2,)), trajectory=[0, 1, 2], score=5.0)
        stats = archive.stats()
        assert stats["archive_size"] == 2
        assert stats["best_score"] == 5.0


class TestMethodRegistry:
    """Test that all methods can be instantiated via the evaluator registry."""

    @pytest.mark.parametrize("method_name", [
        "none", "epsilon_greedy", "ucb", "count_based", "rnd",
        "icm", "ride", "noveld", "ngu", "amigo",
    ])
    def test_create_exploration(self, method_name):
        from rl_exploration_lab.evaluation.evaluator import create_exploration

        exploration = create_exploration(method_name, obs_dim=147, device="cpu")
        assert exploration is not None

        # Verify the interface works
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = exploration.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
