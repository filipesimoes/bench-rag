"""Deterministic offline mock embedding model for testing and offline runs."""

import hashlib
from collections.abc import Callable

import numpy as np

from bench_rag.config import settings
from bench_rag.core.embedding import EmbeddingModel


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic hash-based embedding model with token-overlap sensitivity."""

    def __init__(
        self,
        dimension: int = 1536,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ):
        self._dimension = dimension
        self.query_prefix = (
            query_prefix if query_prefix is not None else settings.embedding_query_prefix
        )
        self.document_prefix = (
            document_prefix if document_prefix is not None else settings.embedding_document_prefix
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed_single(self, text: str) -> list[float]:
        """Produce a normalized vector incorporating character hashes and word tokens."""
        if not text:
            vec = np.zeros(self._dimension, dtype=np.float32)
            vec[0] = 1.0
            return vec.tolist()

        rng = np.random.default_rng(int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:15], 16))
        vec = rng.standard_normal(self._dimension).astype(np.float32)

        # Enhance semantic overlap: add token-derived pseudo-signals
        words = set(text.lower().split())
        for word in words:
            word_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
            idx = word_hash % self._dimension
            vec[idx] += 2.0

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        prefix = self.document_prefix or ""
        results = [self._embed_single(f"{prefix}{t}") for t in texts]
        if progress_callback is not None:
            progress_callback(len(results), len(texts))
        return results

    def embed_query(self, text: str) -> list[float]:
        prefix = self.query_prefix or ""
        return self._embed_single(f"{prefix}{text}")
