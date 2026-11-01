"""OpenAI-compatible embedding model client."""

from collections.abc import Callable
from typing import Any

import httpx

from bench_rag.config import settings
from bench_rag.core.embedding import EmbeddingModel


class OpenAIEmbeddingModel(EmbeddingModel):
    """Client for OpenAI-compatible embedding endpoints (OpenAI, vLLM, Ollama, LocalAI)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        dimension: int | None = None,
        batch_size: int = 64,
        timeout: float = 30.0,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ):
        self.model = model or settings.embedding_model
        resolved_url = base_url or settings.effective_embedding_base_url
        self.base_url = resolved_url.rstrip("/")
        self.api_key = api_key or settings.effective_embedding_api_key
        self._dimension = dimension or settings.embedding_dimension
        self.batch_size = batch_size
        self.timeout = timeout
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

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

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
