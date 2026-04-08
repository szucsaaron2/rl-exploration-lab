"""CLIP encoder wrapper for language-based exploration methods.

Wraps OpenAI's CLIP (or open_clip) to provide:
- Vision encoding: observation image → embedding vector
- Text encoding: language description → embedding vector
- Cosine similarity between vision and text embeddings

Used by: CLIP-RND, CLIP-NovelD, Semantic Exploration, SHELM variants.

If CLIP is not installed, provides a lightweight fallback using random projections
so the library can be used without the heavy transformers dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPEncoder:
    """CLIP encoder for vision-language exploration methods.

    Provides a unified interface for encoding observations and text into
    a shared embedding space. Falls back to a random projection if CLIP
    libraries aren't installed.

    Args:
        model_name: CLIP model name (e.g., 'ViT-B/32').
        device: Torch device.
        embed_dim: Embedding dimension (used for fallback encoder).
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str = "cpu",
        embed_dim: int = 512,
    ):
        self.device = device
        self.embed_dim = embed_dim
        self._clip_available = False

        try:
            import open_clip
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained="openai", device=device
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._model.eval()
            self._clip_available = True
            self.embed_dim = self._model.visual.output_dim
        except (ImportError, Exception):
            # Fallback: random projection encoder
            self._fallback_vision = nn.Linear(147, embed_dim).to(device)
            self._fallback_text = nn.Embedding(10000, embed_dim).to(device)
            # Freeze fallback weights (should be fixed like CLIP)
            for p in self._fallback_vision.parameters():
                p.requires_grad = False
            for p in self._fallback_text.parameters():
                p.requires_grad = False

    @property
    def is_clip_available(self) -> bool:
        return self._clip_available

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode flat MiniGrid observations into CLIP embedding space.

        Args:
            obs: Flat observations, shape (batch, 147) in [0, 1].

        Returns:
            Normalized embeddings, shape (batch, embed_dim).
        """
        if self._clip_available:
            # Reshape flat obs to image-like format for CLIP
            # MiniGrid obs is 7x7x3 — resize to CLIP's expected input
            batch_size = obs.shape[0]
            images = obs.view(batch_size, 7, 7, 3).permute(0, 3, 1, 2)  # (B, 3, 7, 7)
            # Resize to CLIP's expected size (224x224)
            images = F.interpolate(images, size=(224, 224), mode="nearest")
            with torch.no_grad():
                features = self._model.encode_image(images)
            return F.normalize(features.float(), dim=-1)
        else:
            with torch.no_grad():
                features = self._fallback_vision(obs)
            return F.normalize(features, dim=-1)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text descriptions into CLIP embedding space.

        Args:
            texts: List of text strings.

        Returns:
            Normalized embeddings, shape (len(texts), embed_dim).
        """
        if self._clip_available:
            tokens = self._tokenizer(texts).to(self.device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
            return F.normalize(features.float(), dim=-1)
        else:
            # Fallback: hash text to indices and look up embeddings
            indices = torch.tensor(
                [hash(t) % 10000 for t in texts], dtype=torch.long, device=self.device
            )
            with torch.no_grad():
                features = self._fallback_text(indices)
            return F.normalize(features, dim=-1)

    def similarity(self, obs_emb: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between observation and text embeddings.

        Args:
            obs_emb: Observation embeddings, shape (batch, embed_dim).
            text_emb: Text embeddings, shape (n_texts, embed_dim).

        Returns:
            Similarity matrix, shape (batch, n_texts).
        """
        return obs_emb @ text_emb.T
