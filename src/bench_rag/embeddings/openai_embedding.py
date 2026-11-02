"""OpenAI-compatible embedding model client."""

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

from bench_rag.config import settings
from bench_rag.core.embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class OpenAIEmbeddingModel(EmbeddingModel):
    """Client for OpenAI-compatible embedding endpoints (OpenAI, vLLM, Ollama, LocalAI)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 64,
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ):
        self.model = model or settings.embedding_model
        resolved_url = base_url or settings.effective_embedding_base_url
        self.base_url = resolved_url.rstrip("/")
        self.api_key = api_key or settings.effective_embedding_api_key
        self._dimension = dimension or settings.embedding_dimension
        self.batch_size = batch_size
        self.timeout = timeout if timeout is not None else settings.embedding_timeout
        self.max_retries = max_retries if max_retries is not None else settings.http_max_retries
        self.retry_backoff = (
            retry_backoff if retry_backoff is not None else settings.http_retry_backoff
        )
        self.query_prefix = (
            query_prefix if query_prefix is not None else settings.embedding_query_prefix
        )
        self.document_prefix = (
            document_prefix if document_prefix is not None else settings.embedding_document_prefix
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "input": texts,
        }

        data = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                            sleep_s = self.retry_backoff * (2**attempt)
                            logger.warning(
                                "Embedding API call to %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                                url,
                                e.response.status_code,
                                attempt + 1,
                                self.max_retries + 1,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                            continue
                        raise
                    data = response.json()
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    sleep_s = self.retry_backoff * (2**attempt)
                    logger.warning(
                        "Embedding API call to %s failed with %s: %s (attempt %d/%d). Retrying in %.1fs...",
                        url,
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
            raise RuntimeError(f"Embedding API call to {url} failed to return data.")

        data_items = sorted(data["data"], key=lambda x: x["index"])
        embeddings = [item["embedding"] for item in data_items]
        if embeddings and len(embeddings[0]) != self._dimension:
            self._dimension = len(embeddings[0])
        return embeddings

    def embed_texts(
        self,
        texts: list[str],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        prefix = self.document_prefix or ""
        prefixed_texts = [f"{prefix}{t}" for t in texts]

        all_embeddings: list[list[float]] = []
        total = len(prefixed_texts)
        for i in range(0, total, self.batch_size):
            batch = prefixed_texts[i : i + self.batch_size]
            batch_embeddings = self._call_api(batch)
            all_embeddings.extend(batch_embeddings)
            if progress_callback is not None:
                progress_callback(len(all_embeddings), total)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        prefix = self.query_prefix or ""
        prefixed_text = f"{prefix}{text}"
        results = self._call_api([prefixed_text])
        return results[0]
