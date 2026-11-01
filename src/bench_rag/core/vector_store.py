from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from bench_rag.core.chunker import Chunk


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT = "dot"


@dataclass
class SearchResult:
    """Represents a chunk retrieved from a vector database with relevance score."""

    chunk: Chunk
    score: float


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector indexing and approximate nearest neighbor search."""

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        """Create or initialize a collection."""
        ...

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        """Upsert chunks and their corresponding embedding vectors."""
        ...

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for the most similar chunks given a query vector."""
        ...

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        """Retrieve all stored chunks from the collection (useful for hybrid BM25 index)."""
        ...

    def count_chunks(self, collection_name: str) -> int:
        """Return the number of stored chunks/points in the collection."""
        ...
