"""Deterministic mock reranker for testing and dry-run benchmarks."""

from bench_rag.core.reranker import Reranker
from bench_rag.core.vector_store import SearchResult


class MockReranker(Reranker):
    """Mock reranker scoring candidates deterministically without network calls."""

    def __init__(self, model_name: str = "mock-reranker"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        query_tokens = set(query.lower().split())

        scored_items: list[tuple[SearchResult, float, int]] = []
        for original_rank, res in enumerate(results, start=1):
            chunk_tokens = set(res.chunk.text.lower().split())
            overlap = len(query_tokens & chunk_tokens)
            new_score = round(res.score + (overlap * 0.15), 4)
            scored_items.append((res, new_score, original_rank))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        reranked: list[SearchResult] = []
        for res, new_score, orig_rank in scored_items[:top_k]:
            updated_chunk = res.chunk
            updated_metadata = dict(updated_chunk.metadata)
            updated_metadata["initial_retrieval_score"] = res.score
            updated_metadata["initial_retrieval_rank"] = orig_rank
            updated_chunk.metadata = updated_metadata

            reranked.append(SearchResult(chunk=updated_chunk, score=new_score))

        return reranked
