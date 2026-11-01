"""In-memory vector store for testing and lightweight local execution."""

from collections.abc import Callable
from typing import Any

import numpy as np

from bench_rag.core.chunker import Chunk
from bench_rag.core.vector_store import DistanceMetric, SearchResult, VectorStore


class MemoryVectorStore(VectorStore):
    """In-memory numpy vector store supporting cosine, dot, and euclidean distance."""

    def __init__(self):
        self._collections: dict[str, dict[str, Any]] = {}

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        self._collections[collection_name] = {
            "dimension": dimension,
            "metric": metric,
            "chunks": [],
            "vectors": np.empty((0, dimension), dtype=np.float32),
        }

    def add_chunks(
        self,
        collection_name: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        if collection_name not in self._collections:
            dim = len(embeddings[0]) if embeddings else 1536
            self.create_collection(collection_name, dimension=dim)

        col = self._collections[collection_name]
        new_vecs = np.array(embeddings, dtype=np.float32)

        if col["vectors"].shape[0] == 0:
            col["vectors"] = new_vecs
        else:
            col["vectors"] = np.vstack([col["vectors"], new_vecs])

        col["chunks"].extend(chunks)
        if progress_callback is not None:
            progress_callback(len(chunks), len(chunks))

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if collection_name not in self._collections:
            return []

        col = self._collections[collection_name]
        vectors = col["vectors"]
        chunks = col["chunks"]

        if len(chunks) == 0:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        metric = col["metric"]

        if metric == DistanceMetric.COSINE:
            norm_v = np.linalg.norm(vectors, axis=1, keepdims=True)
            norm_q = np.linalg.norm(q_vec)
            norm_v[norm_v == 0] = 1e-10
            if norm_q == 0:
                norm_q = 1e-10
            normed_vectors = vectors / norm_v
            normed_q = q_vec / norm_q
            scores = np.dot(normed_vectors, normed_q)
        elif metric == DistanceMetric.DOT:
            scores = np.dot(vectors, q_vec)
        elif metric == DistanceMetric.EUCLIDEAN:
            # Negative L2 distance so higher is closer
            dists = np.linalg.norm(vectors - q_vec, axis=1)
            scores = -dists
        else:
            scores = np.dot(vectors, q_vec)

        # Apply filter criteria if provided
        filtered_indices = []
        for i, chunk in enumerate(chunks):
            if filter_criteria:
                match = True
                for k, v in filter_criteria.items():
                    if k == "document_id":
                        if chunk.document_id != v:
                            match = False
                            break
                    elif chunk.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_indices.append(i)
            else:
                filtered_indices.append(i)

        if not filtered_indices:
            return []

        filtered_scores = [(idx, float(scores[idx])) for idx in filtered_indices]
        # Sort descending by score
        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for idx, score in filtered_scores[:top_k]:
            results.append(SearchResult(chunk=chunks[idx], score=score))

        return results

    def get_all_chunks(self, collection_name: str) -> list[Chunk]:
        if collection_name not in self._collections:
            return []
        return list(self._collections[collection_name]["chunks"])

    def count_chunks(self, collection_name: str) -> int:
        if collection_name not in self._collections:
            return 0
        return len(self._collections[collection_name]["chunks"])
