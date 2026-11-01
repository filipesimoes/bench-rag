"""Vector store implementations and factory."""

from bench_rag.config import settings
from bench_rag.core.vector_store import VectorStore
from bench_rag.vector_stores.memory_store import MemoryVectorStore
from bench_rag.vector_stores.qdrant_store import QdrantVectorStore


def get_vector_store(
    store_type: str = "memory",
    host: str | None = None,
    port: int | None = None,
    timeout: float | None = None,
) -> VectorStore:
    """Factory function to instantiate a VectorStore."""
    match store_type.lower():
        case "memory":
            return MemoryVectorStore()
        case "qdrant":
            return QdrantVectorStore(
                host=host or settings.qdrant_host,
                port=port or settings.qdrant_port,
                timeout=timeout or settings.qdrant_timeout,
            )
        case _:
            raise ValueError(
                f"Unsupported vector store type: {store_type}. Expected 'memory' or 'qdrant'."
            )


__all__ = ["MemoryVectorStore", "QdrantVectorStore", "get_vector_store"]
