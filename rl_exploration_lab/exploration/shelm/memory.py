"""Semantic HELM memory module (Paischer et al., 2023).

SHELM creates a human-readable memory by:
1. Encoding visual observations through CLIP's vision encoder → z_t
2. Matching z_t against a semantic database S of language token embeddings
3. Retrieving the top-k most similar tokens (human-readable text)
4. Feeding these tokens into a language model (Transformer-XL) as memory

This module implements the retrieval and memory components.
The full Transformer-XL memory is simplified here to focus on the
exploration signal (which is what the thesis investigates).

From the thesis (§3.1.1):
    z_t = CLIP_VM(o_t)
    S* = top-k{cossim(z_t, s) | s ∈ S}
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rl_exploration_lab.networks.encoders import CLIPEncoder


class SemanticDatabase:
    """Semantic database S: pre-computed token embeddings for retrieval.

    The database contains embeddings for a vocabulary of tokens,
    computed by averaging CLIP text embeddings over a set of prompts.

    In the full SHELM: S = {embed(v) | v ∈ V} where V is the intersection
    of CLIP and Transformer-XL vocabularies.

    Simplified version: uses a fixed set of descriptive tokens relevant
    to MiniGrid environments.
    """

    # Tokens relevant to MiniGrid (plus general tokens for comparison)
    DEFAULT_TOKENS = [
        # MiniGrid-relevant
        "wall", "door", "key", "ball", "box", "goal", "lava", "floor", "empty",
        "red", "green", "blue", "purple", "yellow", "grey",
        "open", "closed", "locked", "agent", "corridor", "room",
        "left", "right", "forward", "turn", "pick", "drop",
        # General tokens (to test CLIP's retrieval quality — thesis §5.2)
        "pixel", "square", "miner", "pong", "game", "grid",
        "navigate", "explore", "object", "color", "position",
        "small", "large", "near", "far", "inside", "outside",
    ]

    def __init__(self, clip_encoder: CLIPEncoder, tokens: list[str] | None = None):
        self.clip = clip_encoder
        self.tokens = tokens or self.DEFAULT_TOKENS
        self._embeddings: torch.Tensor | None = None

    def build(self) -> None:
        """Pre-compute token embeddings using CLIP's text encoder."""
        self._embeddings = self.clip.encode_text(self.tokens)

    @property
    def embeddings(self) -> torch.Tensor:
        if self._embeddings is None:
            self.build()
        return self._embeddings

    def retrieve_top_k(
        self, obs_embedding: torch.Tensor, k: int = 4
    ) -> tuple[list[list[str]], torch.Tensor]:
        """Retrieve the top-k most similar tokens for each observation.

        Args:
            obs_embedding: CLIP vision embeddings, shape (batch, embed_dim).
            k: Number of tokens to retrieve per observation.

        Returns:
            tokens: List of token lists (one per batch element).
            embeddings: Corresponding token embeddings, shape (batch, k, embed_dim).
        """
        # Cosine similarity between observation and all tokens
        similarities = obs_embedding @ self.embeddings.T  # (batch, n_tokens)
        topk_values, topk_indices = similarities.topk(k, dim=-1)

        batch_tokens = []
        batch_embeddings = []

        for i in range(obs_embedding.shape[0]):
            tokens = [self.tokens[idx] for idx in topk_indices[i].tolist()]
            embs = self.embeddings[topk_indices[i]]  # (k, embed_dim)
            batch_tokens.append(tokens)
            batch_embeddings.append(embs)

        return batch_tokens, torch.stack(batch_embeddings)


class SHELMMemory(nn.Module):
    """Simplified SHELM memory module for exploration.

    Instead of the full Transformer-XL memory, this uses the retrieved
    top-k token embeddings directly (or their average) as the memory
    representation. This matches the thesis Phase 2 experiments (§5.2).

    Three output modes (matching thesis experiments):
    - 'embeddings': use the top-k token embeddings directly
    - 'tokens': encode decoded token strings back through CLIP
    - 'average': average the top-k token embeddings

    Args:
        clip_encoder: CLIP encoder instance.
        top_k: Number of tokens to retrieve.
        output_mode: How to produce the memory output.
    """

    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        top_k: int = 4,
        output_mode: str = "embeddings",
    ):
        super().__init__()
        self.clip = clip_encoder
        self.top_k = top_k
        self.output_mode = output_mode
        self.semantic_db = SemanticDatabase(clip_encoder)
        self.semantic_db.build()

    @property
    def output_dim(self) -> int:
        if self.output_mode == "average":
            return self.clip.embed_dim
        elif self.output_mode == "embeddings":
            return self.clip.embed_dim * self.top_k
        elif self.output_mode == "tokens":
            return self.clip.embed_dim
        else:
            return self.clip.embed_dim

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, list[list[str]]]:
        """Process observations through SHELM's memory pipeline.

        Args:
            obs: Flat observations, shape (batch, obs_dim).

        Returns:
            memory_output: Memory representation, shape depends on output_mode.
            retrieved_tokens: Human-readable tokens retrieved for each observation.
        """
        # Step 1: Encode observations through CLIP vision encoder
        obs_embedding = self.clip.encode_observation(obs)

        # Step 2: Retrieve top-k tokens from semantic database
        tokens, token_embeddings = self.semantic_db.retrieve_top_k(obs_embedding, self.top_k)
        # token_embeddings shape: (batch, top_k, embed_dim)

        # Step 3: Produce memory output based on mode
        if self.output_mode == "average":
            memory = token_embeddings.mean(dim=1)  # (batch, embed_dim)
        elif self.output_mode == "embeddings":
            memory = token_embeddings.view(obs.shape[0], -1)  # (batch, top_k * embed_dim)
        elif self.output_mode == "tokens":
            # Re-encode the retrieved token strings through CLIP
            # (This is the "decoded token strings" variant from thesis §5.2)
            all_tokens = [" ".join(t) for t in tokens]
            memory = self.clip.encode_text(all_tokens)  # (batch, embed_dim)
        else:
            memory = token_embeddings.mean(dim=1)

        return memory, tokens
