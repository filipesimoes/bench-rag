"""Embedding implementations and factory."""

from bench_rag.core.embedding import EmbeddingModel
from bench_rag.embeddings.mock_embedding import MockEmbeddingModel
from bench_rag.embeddings.openai_embedding import OpenAIEmbeddingModel


def get_embedding_model(
    embedding_type: str = "mock",
    model_name: str | None = None,
    dimension: int = 1536,
    base_url: str | None = None,
    api_key: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> EmbeddingModel:
    """Factory function to instantiate an EmbeddingModel."""
    match embedding_type.lower():
        case "mock":
            return MockEmbeddingModel(
                dimension=dimension,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
            )
        case "openai":
            return OpenAIEmbeddingModel(
                model=model_name,
                dimension=dimension,
                base_url=base_url,
                api_key=api_key,
                query_prefix=query_prefix,
                document_prefix=document_prefix,
                timeout=timeout,
                max_retries=max_retries,
            )
        case _:
            raise ValueError(
                f"Unsupported embedding type: {embedding_type}. Expected 'mock' or 'openai'."
            )


__all__ = ["MockEmbeddingModel", "OpenAIEmbeddingModel", "get_embedding_model"]
