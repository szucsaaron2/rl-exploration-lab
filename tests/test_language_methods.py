"""Tests for language-based exploration methods (Phase 3)."""

import pytest
import torch
import numpy as np


class TestLanguageOracle:
    def test_describe_flat_obs(self):
        from rl_exploration_lab.envs.language_oracle import LanguageOracle

        oracle = LanguageOracle(verbose=True)
        obs = np.random.rand(147).astype(np.float32)
        desc = oracle.describe_observation(obs)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_describe_concise(self):
        from rl_exploration_lab.envs.language_oracle import LanguageOracle

        oracle = LanguageOracle(verbose=False)
        obs = np.random.rand(147).astype(np.float32)
        desc = oracle.describe_observation(obs)
        assert isinstance(desc, str)

    def test_describe_with_direction_and_carrying(self):
        from rl_exploration_lab.envs.language_oracle import LanguageOracle

        oracle = LanguageOracle(verbose=True)
        obs = np.random.rand(147).astype(np.float32)
        desc = oracle.describe_observation(obs, agent_dir=0, carrying="key")
        assert "right" in desc.lower()
        assert "key" in desc.lower()

    def test_describe_full_state(self):
        from rl_exploration_lab.envs.language_oracle import LanguageOracle

        oracle = LanguageOracle()
        obs = np.random.rand(147).astype(np.float32)
        desc = oracle.describe_full_state(obs, agent_pos=(3, 5), agent_dir=1)
        assert "(3, 5)" in desc


class TestCLIPEncoder:
    def test_encode_observation(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder

        clip = CLIPEncoder(device="cpu")
        obs = torch.randn(4, 147)
        emb = clip.encode_observation(obs)
        assert emb.shape == (4, clip.embed_dim)
        # Should be normalized
        norms = emb.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_text(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder

        clip = CLIPEncoder(device="cpu")
        texts = ["a red key", "an open door", "empty room"]
        emb = clip.encode_text(texts)
        assert emb.shape == (3, clip.embed_dim)

    def test_similarity(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder

        clip = CLIPEncoder(device="cpu")
        obs_emb = clip.encode_observation(torch.randn(2, 147))
        text_emb = clip.encode_text(["key", "door"])
        sim = clip.similarity(obs_emb, text_emb)
        assert sim.shape == (2, 2)


class TestSHELMMemory:
    def test_forward_embeddings_mode(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder
        from rl_exploration_lab.exploration.shelm.memory import SHELMMemory

        clip = CLIPEncoder(device="cpu")
        shelm = SHELMMemory(clip, top_k=4, output_mode="embeddings")
        obs = torch.randn(4, 147)
        memory, tokens = shelm(obs)
        assert memory.shape[0] == 4
        assert memory.shape[1] == clip.embed_dim * 4  # top_k * embed_dim
        assert len(tokens) == 4
        assert len(tokens[0]) == 4  # top_k tokens per obs

    def test_forward_average_mode(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder
        from rl_exploration_lab.exploration.shelm.memory import SHELMMemory

        clip = CLIPEncoder(device="cpu")
        shelm = SHELMMemory(clip, top_k=4, output_mode="average")
        obs = torch.randn(4, 147)
        memory, tokens = shelm(obs)
        assert memory.shape == (4, clip.embed_dim)

    def test_forward_tokens_mode(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder
        from rl_exploration_lab.exploration.shelm.memory import SHELMMemory

        clip = CLIPEncoder(device="cpu")
        shelm = SHELMMemory(clip, top_k=2, output_mode="tokens")
        obs = torch.randn(4, 147)
        memory, tokens = shelm(obs)
        assert memory.shape == (4, clip.embed_dim)

    def test_retrieved_tokens_are_strings(self):
        from rl_exploration_lab.networks.encoders import CLIPEncoder
        from rl_exploration_lab.exploration.shelm.memory import SHELMMemory

        clip = CLIPEncoder(device="cpu")
        shelm = SHELMMemory(clip, top_k=3, output_mode="average")
        obs = torch.randn(2, 147)
        _, tokens = shelm(obs)
        for batch_tokens in tokens:
            for t in batch_tokens:
                assert isinstance(t, str)


class TestCLIPRND:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.language.clip_rnd import CLIPRND

        clip_rnd = CLIPRND(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = clip_rnd.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
        assert (reward >= 0).all()

    def test_update(self):
        from rl_exploration_lab.exploration.language.clip_rnd import CLIPRND

        clip_rnd = CLIPRND(obs_dim=147)
        metrics = clip_rnd.update({"next_obs": torch.randn(8, 147)})
        assert "exploration_loss" in metrics


class TestCLIPNovelD:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.language.clip_noveld import CLIPNovelD

        clip_nd = CLIPNovelD(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = clip_nd.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
        assert (reward >= 0).all()


class TestSemanticExploration:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.language.semantic import SemanticExploration

        se = SemanticExploration(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = se.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
        assert (reward >= 0).all()


class TestLNovelD:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.language.l_noveld import LNovelD

        l_nd = LNovelD(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = l_nd.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
        assert (reward >= 0).all()


class TestLAMIGo:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.language.l_amigo import LAMIGo

        l_amigo = LAMIGo(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = l_amigo.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)


class TestSHELMRND:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.shelm.shelm_rnd import SHELMRND

        shelm_rnd = SHELMRND(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = shelm_rnd.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
        assert (reward >= 0).all()

    def test_update(self):
        from rl_exploration_lab.exploration.shelm.shelm_rnd import SHELMRND

        shelm_rnd = SHELMRND(obs_dim=147)
        metrics = shelm_rnd.update({"next_obs": torch.randn(8, 147)})
        assert "exploration_loss" in metrics

    def test_tokens_returned(self):
        from rl_exploration_lab.exploration.shelm.shelm_rnd import SHELMRND

        shelm_rnd = SHELMRND(obs_dim=147, top_k=3)
        obs = torch.randn(2, 147)
        shelm_rnd.compute_intrinsic_reward(obs, obs, torch.zeros(2, dtype=torch.long))
        tokens = shelm_rnd.get_last_tokens()
        assert len(tokens) == 2
        assert len(tokens[0]) == 3


class TestSHELMOracle:
    def test_reward_shape(self):
        from rl_exploration_lab.exploration.shelm.shelm_oracle import SHELMOracle

        shelm_oracle = SHELMOracle(obs_dim=147)
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = shelm_oracle.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
        assert (reward >= 0).all()

    def test_descriptions_returned(self):
        from rl_exploration_lab.exploration.shelm.shelm_oracle import SHELMOracle

        shelm_oracle = SHELMOracle(obs_dim=147)
        obs = torch.randn(2, 147)
        shelm_oracle.compute_intrinsic_reward(obs, obs, torch.zeros(2, dtype=torch.long))
        descs = shelm_oracle.get_last_descriptions()
        assert len(descs) == 2
        assert all(isinstance(d, str) for d in descs)

    def test_update(self):
        from rl_exploration_lab.exploration.shelm.shelm_oracle import SHELMOracle

        shelm_oracle = SHELMOracle(obs_dim=147)
        metrics = shelm_oracle.update({"next_obs": torch.randn(8, 147)})
        assert "exploration_loss" in metrics


class TestFullRegistry:
    """Test ALL 17 methods can be created and produce valid rewards."""

    @pytest.mark.parametrize("method_name", [
        "none", "epsilon_greedy", "ucb", "count_based", "rnd",
        "icm", "ride", "noveld", "ngu", "amigo",
        "clip_rnd", "clip_noveld", "semantic", "l_noveld", "l_amigo",
        "shelm_rnd", "shelm_oracle",
    ])
    def test_create_and_compute_reward(self, method_name):
        from rl_exploration_lab.evaluation.evaluator import create_exploration

        exploration = create_exploration(method_name, obs_dim=147, device="cpu")
        obs = torch.randn(4, 147)
        next_obs = torch.randn(4, 147)
        action = torch.randint(0, 7, (4,))
        reward = exploration.compute_intrinsic_reward(obs, next_obs, action)
        assert reward.shape == (4,)
