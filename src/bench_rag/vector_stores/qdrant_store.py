"""Qdrant vector store implementation."""

import uuid
from collections.abc import Callable
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from bench_rag.config import settings
from bench_rag.core.chunker import Chunk
from bench_rag.core.vector_store import DistanceMetric, SearchResult, VectorStore


class QdrantVectorStore(VectorStore):
    """Vector database implementation using Qdrant client."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        timeout: float | None = None,
        url: str | None = None,
        api_key: str | None = None,
        location: str | None = None,
        client: QdrantClient | None = None,
    ):
        self.host = host or settings.qdrant_host
        self.port = port or settings.qdrant_port
        self.timeout = timeout if timeout is not None else settings.qdrant_timeout

        if client is not None:
            self.client = client
        elif location is not None:
            self.client = QdrantClient(
                location=location,
                api_key=api_key,
                check_compatibility=False,
                timeout=self.timeout,
            )
        elif url:
            self.client = QdrantClient(
                url=url,
                api_key=api_key,
                check_compatibility=False,
                timeout=self.timeout,
            )
        else:
            self.client = QdrantClient(
                host=self.host,
                port=self.port,
                api_key=api_key,
                check_compatibility=False,
                timeout=self.timeout,
            )

    def _convert_distance_metric(self, metric: DistanceMetric) -> models.Distance:
        match metric:
            case DistanceMetric.COSINE:
                return models.Distance.COSINE
            case DistanceMetric.EUCLIDEAN:
                return models.Distance.EUCLID
            case DistanceMetric.DOT:
                return models.Distance.DOT
            case _:
                return models.Distance.COSINE

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        collections = self.client.get_collections().collections
        existing_names = [col.name for col in collections]

        if collection_name not in existing_names:
            qdrant_distance = self._convert_distance_metric(metric)
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=qdrant_distance,
                ),
            )

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if not chunks:
            return

        points: list[models.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "token_count": chunk.token_count,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upload
        batch_size = 100
        total = len(points)
        for i in range(0, total, batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
            if progress_callback is not None:
                progress_callback(min(i + batch_size, total), total)

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query_filter = None
        if filter_criteria:
            conditions: list[models.FieldCondition] = []
            for k, v in filter_criteria.items():
                conditions.append(
                    models.FieldCondition(
                        key=k,
                        match=models.MatchValue(value=v),
                    )
                )
            if conditions:
                query_filter = models.Filter(must=conditions)

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )
            points = response.points
        else:
            points = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )

        results: list[SearchResult] = []
        for hit in points:
            payload = hit.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id", str(hit.id)),
                document_id=payload.get("document_id", ""),
                text=payload.get("text", ""),
                metadata=payload.get("metadata", {}),
                start_char=payload.get("start_char", 0),
                end_char=payload.get("end_char", 0),
                token_count=payload.get("token_count", 0),
            )
            results.append(SearchResult(chunk=chunk, score=float(hit.score)))

        return results

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        """Scroll all points to retrieve chunks for hybrid BM25 indexing."""
        chunks: list[Chunk] = []
        offset = None

        while True:
            records, offset = self.client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                payload = rec.payload or {}
                chunk = Chunk(
                    id=payload.get("chunk_id", str(rec.id)),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    metadata=payload.get("metadata", {}),
                    start_char=payload.get("start_char", 0),
                    end_char=payload.get("end_char", 0),
                    token_count=payload.get("token_count", 0),
                )
                chunks.append(chunk)

            if offset is None:
                break

        return chunks

    def count_chunks(self, collection_name: str) -> int:
        if hasattr(self.client, "collection_exists") and not self.client.collection_exists(
            collection_name
        ):
            return 0
        try:
            res = self.client.count(collection_name=collection_name)
            return res.count
        except Exception:  # noqa: BLE001
            return 0
