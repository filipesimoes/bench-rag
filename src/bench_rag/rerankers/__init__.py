"""Reranker module implementations and factory."""

from bench_rag.core.reranker import Reranker
from bench_rag.rerankers.http_reranker import HttpReranker
from bench_rag.rerankers.mock_reranker import MockReranker


def get_reranker(
    reranker_type: str = "none",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Reranker | None:
    """Instantiate a Reranker based on provider type."""
    t = (reranker_type or "none").lower()
    if t == "none":
        return None
    elif t == "mock":
        return MockReranker(model_name=model or "mock-reranker")
    elif t in ("http", "vllm", "tei", "cohere", "jina"):
        return HttpReranker(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown reranker_type: '{reranker_type}'")


__all__ = [
    "HttpReranker",
    "MockReranker",
    "Reranker",
    "get_reranker",
]
