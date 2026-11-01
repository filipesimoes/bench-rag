"""Core EmbeddingModel protocol."""

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for vector embedding models."""

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Generate vector embeddings for a list of texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate vector embedding for a single search query."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimension of the embedding model."""
        ...
