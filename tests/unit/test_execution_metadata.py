"""Unit tests for pipeline execution metadata extraction."""

from bench_rag.config import PipelineConfig
from bench_rag.core.evaluator import LLMJudgeMetrics
from bench_rag.core.generator import GenerationResult, GeneratorLLM
from bench_rag.embeddings.mock_embedding import MockEmbeddingModel
from bench_rag.evaluation.metadata import extract_execution_metadata
from bench_rag.evaluation.metrics.llm_judge import LLMJudge
from bench_rag.generators.mock_generator import MockGeneratorLLM
from bench_rag.pipelines.hybrid_rag import HybridRAGPipeline
from bench_rag.pipelines.standard_rag import StandardDenseRAGPipeline
from bench_rag.rerankers.mock_reranker import MockReranker
from bench_rag.vector_stores.memory_store import MemoryVectorStore


def test_extract_execution_metadata_with_config():
    cfg = PipelineConfig(
        name="test_standard_pipeline",
        pipeline_type="standard",
        top_k=7,
        generator_type="mock",
        generator_model="custom-generator-v1",
        embedding_type="mock",
        embedding_model="custom-embedding-v1",
        reranker_type="mock",
        reranker_model="custom-reranker-v1",
        judge_type="mock",
        judge_model="custom-judge-v1",
    )

    meta = extract_execution_metadata(config=cfg, benchmark_csv="data/test_ground_truth.csv", total_samples=10)

    assert meta["pipeline_name"] == "test_standard_pipeline"
    assert meta["pipeline_type"] == "Standard Dense RAG"
    assert meta["top_k"] == 7
    assert meta["generator_model"] == "custom-generator-v1"
    assert meta["embedding_model"] == "custom-embedding-v1"
    assert meta["reranker_model"] == "custom-reranker-v1"
    assert meta["judge_model"] == "custom-judge-v1"
    assert meta["benchmark_csv"] == "data/test_ground_truth.csv"
    assert meta["total_samples"] == 10

    # Ensure no API keys or sensitive credentials leak
    assert "api_key" not in meta
    for k in meta:
        assert "key" not in k.lower()


def test_extract_execution_metadata_hybrid_and_no_reranker():
    cfg = PipelineConfig(
        name="test_hybrid_pipeline",
        pipeline_type="hybrid",
        top_k=3,
        reranker_type="none",
    )

    meta = extract_execution_metadata(config=cfg)

    assert meta["pipeline_name"] == "test_hybrid_pipeline"
    assert meta["pipeline_type"] == "Hybrid BM25 + Dense RAG"
    assert meta["top_k"] == 3
    assert meta["reranker_model"] == "None (Direct Retrieval)"


def test_extract_execution_metadata_from_pipeline_instances():
    embed = MockEmbeddingModel(dimension=128)
    store = MemoryVectorStore()
    gen = MockGeneratorLLM(model_name="mock-gen-alpha")
    rerank = MockReranker(model_name="mock-rerank-beta")
    judge_gen = MockGeneratorLLM(model_name="mock-judge-gamma")
    judge = LLMJudge(generator=judge_gen)

    dense_pipe = StandardDenseRAGPipeline(
        name="dense_active_pipeline",
        embedding_model=embed,
        vector_store=store,
        generator=gen,
        reranker=rerank,
        top_k=4,
    )

    meta = extract_execution_metadata(pipeline=dense_pipe, judge=judge, top_k=4)

    assert meta["pipeline_name"] == "dense_active_pipeline"
    assert meta["pipeline_type"] == "Standard Dense RAG"
    assert meta["top_k"] == 4
    assert meta["generator_model"] == "mock-gen-alpha"
    assert meta["reranker_model"] == "mock-rerank-beta"
    assert meta["judge_model"] == "mock-judge-gamma"


def test_extract_execution_metadata_defaults():
    meta = extract_execution_metadata()

    assert meta["pipeline_name"] == "standard_dense_rag"
    assert meta["pipeline_type"] == "Standard Dense RAG"
    assert meta["top_k"] == 5
    assert meta["reranker_model"] == "None (Direct Retrieval)"
