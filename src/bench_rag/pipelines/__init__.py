"""RAG Pipeline implementations and factory."""

from bench_rag.config import PipelineConfig
from bench_rag.core.pipeline import RAGPipeline
from bench_rag.embeddings import get_embedding_model
from bench_rag.generators import get_generator
from bench_rag.pipelines.hybrid_rag import HybridRAGPipeline
from bench_rag.pipelines.standard_rag import StandardDenseRAGPipeline
from bench_rag.rerankers import get_reranker
from bench_rag.vector_stores import get_vector_store


def build_pipeline_from_config(config: PipelineConfig) -> RAGPipeline:
    """Build and wire an end-to-end RAGPipeline from a PipelineConfig specification."""
    embedding = get_embedding_model(
        embedding_type=config.embedding_type,
        model_name=config.embedding_model,
        base_url=config.embedding_base_url,
        query_prefix=config.embedding_query_prefix,
        document_prefix=config.embedding_document_prefix,
        timeout=config.embedding_timeout,
        max_retries=config.max_retries,
    )
    vector_store = get_vector_store(
        store_type=config.vector_store_type,
    )
    generator = get_generator(
        generator_type=config.generator_type,
        model_name=config.generator_model,
        base_url=config.generator_base_url,
        enable_thinking=config.generator_enable_thinking,
        dialect=config.generator_dialect,
        max_tokens=config.generator_max_tokens,
        temperature=config.generator_temperature,
        timeout=config.generator_timeout,
        max_retries=config.max_retries,
        response_format=config.generator_response_format,
    )
    reranker = get_reranker(
        reranker_type=config.reranker_type,
        model=config.reranker_model,
        base_url=config.reranker_base_url,
        timeout=config.reranker_timeout,
        max_retries=config.max_retries,
    )

    if config.pipeline_type == "hybrid":
        return HybridRAGPipeline(
            name=config.name,
            embedding_model=embedding,
            vector_store=vector_store,
            generator=generator,
            reranker=reranker,
            rerank_fetch_k=config.rerank_fetch_k,
            top_k=config.top_k,
            collection_name=config.collection_name,
            rrf_k=config.rrf_k,
            dense_weight=config.dense_weight,
            bm25_weight=config.bm25_weight,
            system_prompt=config.system_prompt,
        )
    else:
        return StandardDenseRAGPipeline(
            name=config.name,
            embedding_model=embedding,
            vector_store=vector_store,
            generator=generator,
            reranker=reranker,
            rerank_fetch_k=config.rerank_fetch_k,
            top_k=config.top_k,
            collection_name=config.collection_name,
            system_prompt=config.system_prompt,
        )


__all__ = [
    "HybridRAGPipeline",
    "StandardDenseRAGPipeline",
    "build_pipeline_from_config",
]
