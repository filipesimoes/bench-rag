"""Universal HTTP client for vLLM, TEI and OpenAPI compatible /v1/rerank endpoints."""

import logging
import time
from typing import Any

import httpx

from bench_rag.config import settings
from bench_rag.core.reranker import Reranker
from bench_rag.core.vector_store import SearchResult

logger = logging.getLogger(__name__)


class HttpReranker(Reranker):
    """Client for vLLM, TEI and OpenAPI compatible rerank endpoints."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
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
        self.timeout = timeout if timeout is not None else settings.reranker_timeout
        self.max_retries = max_retries if max_retries is not None else settings.http_max_retries
        self.retry_backoff = (
            retry_backoff if retry_backoff is not None else settings.http_retry_backoff
        )

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

        data = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.endpoint, json=payload, headers=headers)
                    try:
                        resp.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                            sleep_s = self.retry_backoff * (2**attempt)
                            logger.warning(
                                "Reranker API call to %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                                self.endpoint,
                                e.response.status_code,
                                attempt + 1,
                                self.max_retries + 1,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                            continue
                        raise
                    data = resp.json()
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    sleep_s = self.retry_backoff * (2**attempt)
                    logger.warning(
                        "Reranker API call to %s failed with %s: %s (attempt %d/%d). Retrying in %.1fs...",
                        self.endpoint,
                        type(e).__name__,
                        str(e),
                        attempt + 1,
                        self.max_retries + 1,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise

        if data is None:
            raise RuntimeError(f"Reranker API call to {self.endpoint} failed to return data.")

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
