"""Embedders for memory retrieval.

HashEmbedder   — deterministic, no dependencies, fast but low quality.
STEmbedder     — sentence-transformers, semantically meaningful, recommended
                 for paper runs. Requires: pip install sentence-transformers
"""
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Structural interface shared by HashEmbedder and STEmbedder."""

    dim: int

    def embed(self, text: str) -> np.ndarray: ...
    def embed_to_list(self, text: str) -> list[float]: ...


class HashEmbedder:
    def __init__(self, vocab_size: int = 4096, dim: int = 256, seed: int = 42):
        self.vocab_size = vocab_size
        self.dim = dim
        rng = np.random.RandomState(seed)
        proj = rng.randn(vocab_size, dim).astype(np.float32)
        col_norms = np.linalg.norm(proj, axis=0, keepdims=True)
        self.projection: np.ndarray = proj / (col_norms + 1e-8)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _hash_token(self, token: str) -> int:
        # FNV-1a, folded to vocab_size
        h = 2166136261
        for c in token.encode():
            h ^= c
            h = (h * 16777619) & 0xFFFFFFFF
        return h % self.vocab_size

    def embed(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        bow = np.zeros(self.vocab_size, dtype=np.float32)
        for t in tokens:
            bow[self._hash_token(t)] += 1.0
        bow /= len(tokens)
        vec = bow @ self.projection
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def embed_to_list(self, text: str) -> list[float]:
        return self.embed(text).tolist()


class STEmbedder:
    """Sentence-transformers embedder.

    Semantically meaningful dense embeddings. Required for paper-quality
    retrieval — the hash embedder cannot distinguish ScienceWorld states
    well enough for Q-values to spread beyond near-zero.

    First use downloads the model (~22 MB for all-MiniLM-L6-v2) and caches
    it locally. Subsequent runs are instant.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers required. "
                "Run: pip install sentence-transformers"
            ) from exc
        self._model: Any = SentenceTransformer(model_name)  # type: ignore[operator]
        # dim is determined by the loaded model
        probe: np.ndarray = self._model.encode(["probe"], convert_to_numpy=True)  # type: ignore[reportUnknownMemberType]
        self.dim: int = int(probe.shape[1])  # type: ignore[reportUnknownMemberType]

    def embed(self, text: str) -> np.ndarray:
        vec: np.ndarray = self._model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        return vec.astype(np.float32)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed multiple texts at once — much faster than calling embed() in a loop."""
        return self._model.encode(  # type: ignore[return-value]
            texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    def embed_to_list(self, text: str) -> list[float]:
        return self.embed(text).tolist()
