"""Extraction of essential execution parameters and model metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bench_rag.config import settings

if TYPE_CHECKING:
    from bench_rag.config import PipelineConfig
    from bench_rag.core.pipeline import RAGPipeline


def extract_execution_metadata(
    config: PipelineConfig | None = None,
    pipeline: RAGPipeline | None = None,
    judge: Any | None = None,
    top_k: int | None = None,
    benchmark_csv: str | None = None,
    total_samples: int | None = None,
) -> dict[str, Any]:
    """Extract essential model identifiers and retrieval parameters for reporting.

    Omits provider details and sensitive credentials, recording only effective
    model names and core pipeline execution settings.
    """
    # 1. Pipeline Name
    if config and getattr(config, "name", None):
        p_name = config.name
    elif pipeline and getattr(pipeline, "name", None):
        p_name = pipeline.name
    else:
        p_name = "standard_dense_rag"

    # 2. Pipeline Architecture
    if config and getattr(config, "pipeline_type", None):
        p_type = (
            "Hybrid BM25 + Dense RAG"
            if config.pipeline_type == "hybrid"
            else "Standard Dense RAG"
        )
    elif pipeline:
        cls_name = type(pipeline).__name__.lower()
        name_lower = getattr(pipeline, "name", "").lower()
        if "hybrid" in cls_name or "hybrid" in name_lower:
            p_type = "Hybrid BM25 + Dense RAG"
        else:
            p_type = "Standard Dense RAG"
    else:
        p_type = "Standard Dense RAG"

    # 3. Top-K Chunks
    if top_k is not None:
        effective_top_k = top_k
    elif config and getattr(config, "top_k", None) is not None:
        effective_top_k = config.top_k
    elif pipeline and getattr(pipeline, "top_k", None) is not None:
        effective_top_k = pipeline.top_k
    else:
        effective_top_k = 5

    # 4. Generator Model
    generator_model = None
    gen_obj = getattr(pipeline, "generator", None)
    if gen_obj is not None:
        generator_model = getattr(gen_obj, "model", None) or getattr(
            gen_obj, "model_name", None
        )
    if not generator_model:
        if config:
            if getattr(config, "generator_type", "mock") == "mock":
                generator_model = config.generator_model or "mock-generator"
            else:
                generator_model = config.generator_model or settings.generator_model
        else:
            generator_model = settings.generator_model

    # 5. Embedding Model
    embedding_model = None
    emb_obj = getattr(pipeline, "embedding_model", None)
    if emb_obj is not None:
        embedding_model = getattr(emb_obj, "model", None) or getattr(
            emb_obj, "model_name", None
        )
    if not embedding_model:
        if config:
            if getattr(config, "embedding_type", "mock") == "mock":
                embedding_model = config.embedding_model or "mock-embedding"
            else:
                embedding_model = config.embedding_model or settings.embedding_model
        else:
            embedding_model = settings.embedding_model

    # 6. Reranker Model
    reranker_model = "None (Direct Retrieval)"
    if pipeline is not None and hasattr(pipeline, "reranker"):
        reranker_obj = pipeline.reranker
        if reranker_obj is None:
            reranker_model = "None (Direct Retrieval)"
        else:
            reranker_model = getattr(reranker_obj, "model", None) or getattr(
                reranker_obj, "model_name", None
            ) or "Active Reranker"
    elif config:
        t = (getattr(config, "reranker_type", "none") or "none").lower()
        if t in ("none", ""):
            reranker_model = "None (Direct Retrieval)"
        elif t == "mock":
            reranker_model = config.reranker_model or "mock-reranker"
        else:
            reranker_model = config.reranker_model or settings.effective_reranker_model

    # 7. LLM Judge Model
    judge_model = "N/A"
    if judge is not None:
        judge_gen = getattr(judge, "generator", None)
        if judge_gen is not None:
            judge_model = getattr(judge_gen, "model", None) or getattr(
                judge_gen, "model_name", None
            ) or "N/A"
        else:
            judge_model = getattr(judge, "model", None) or getattr(
                judge, "model_name", None
            ) or "N/A"
    elif config:
        if getattr(config, "judge_type", "mock") == "mock":
            judge_model = config.judge_model or "mock-judge"
        else:
            judge_model = config.judge_model or settings.judge_model
    else:
        judge_model = settings.judge_model

    metadata: dict[str, Any] = {
        "pipeline_name": p_name,
        "pipeline_type": p_type,
        "top_k": effective_top_k,
        "generator_model": generator_model,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "judge_model": judge_model,
    }

    if benchmark_csv:
        metadata["benchmark_csv"] = str(benchmark_csv)
    if total_samples is not None:
        metadata["total_samples"] = total_samples

    return metadata
