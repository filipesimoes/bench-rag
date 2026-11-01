"""Core interfaces and abstractions for bench-rag."""

from bench_rag.core.chunker import Chunk, Chunker
from bench_rag.core.cleaner import DocumentCleaner
from bench_rag.core.embedding import EmbeddingModel
from bench_rag.core.evaluator import (
    EvaluationEngine,
    EvaluationSummary,
    GroundTruthItem,
    LexicalMetrics,
    LLMJudgeMetrics,
    RetrievalMetrics,
    SampleEvaluation,
    TaxonomyAggregate,
)
from bench_rag.core.generator import GenerationResult, GeneratorLLM
from bench_rag.core.parser import Document, DocumentParser
from bench_rag.core.pipeline import RAGPipeline, RAGResponse
from bench_rag.core.reranker import Reranker
from bench_rag.core.vector_store import DistanceMetric, SearchResult, VectorStore

__all__ = [
    "Chunk",
    "Chunker",
    "DistanceMetric",
    "Document",
    "DocumentCleaner",
    "DocumentParser",
    "EmbeddingModel",
    "EvaluationEngine",
    "EvaluationSummary",
    "GenerationResult",
    "GeneratorLLM",
    "GroundTruthItem",
    "LLMJudgeMetrics",
    "LexicalMetrics",
    "RAGPipeline",
    "RAGResponse",
    "Reranker",
    "RetrievalMetrics",
    "SampleEvaluation",
    "SearchResult",
    "TaxonomyAggregate",
    "VectorStore",
]
