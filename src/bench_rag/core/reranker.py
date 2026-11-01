"""Core Reranker protocol for second-stage candidate re-ranking."""

from typing import Protocol, runtime_checkable

from bench_rag.core.vector_store import SearchResult


@runtime_checkable
class Reranker(Protocol):
    """Protocol for scoring and reordering candidate chunks retrieved in the first stage."""

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Rerank candidate SearchResults and return the top_k most relevant chunks."""
        ...
