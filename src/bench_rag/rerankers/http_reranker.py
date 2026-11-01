"""Universal HTTP client for vLLM, TEI and OpenAPI compatible /v1/rerank endpoints."""

from typing import Any

import httpx

from bench_rag.config import settings
from bench_rag.core.reranker import Reranker
from bench_rag.core.vector_store import SearchResult


class HttpReranker(Reranker):
    """Client for vLLM, TEI and OpenAPI compatible rerank endpoints."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or settings.effective_reranker_model
        resolved_url = (base_url or settings.effective_reranker_base_url).rstrip("/")

        # Standardize endpoint resolution for vLLM / TEI / Cohere
        if resolved_url.endswith("/rerank"):
            self.endpoint = resolved_url
        elif resolved_url.endswith("/v1"):
            self.endpoint = f"{resolved_url}/rerank"
        else:
            self.endpoint = f"{resolved_url}/v1/rerank"

        self.api_key = api_key or settings.effective_reranker_api_key
        self.timeout = timeout

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key and self.api_key != "mock-key":
            headers["Authorization"] = f"Bearer {self.api_key}"

        docs = [res.chunk.text for res in results]
        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": min(top_k, len(results)),
        }

        # Fail-fast: raise_for_status without catching to prevent silent degradation
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        raw_results = data.get("results", data)
        reranked: list[SearchResult] = []

        for item in raw_results:
            idx = item["index"]
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            original_res = results[idx]

            updated_chunk = original_res.chunk
            updated_metadata = dict(updated_chunk.metadata)
            updated_metadata["initial_retrieval_score"] = original_res.score
            updated_metadata["initial_retrieval_rank"] = idx + 1
            updated_chunk.metadata = updated_metadata

            reranked.append(SearchResult(chunk=updated_chunk, score=score))

        return reranked
