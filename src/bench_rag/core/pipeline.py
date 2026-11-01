"""Core RAGPipeline protocol and RAGResponse dataclass."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from bench_rag.core.vector_store import SearchResult


@dataclass
class RAGResponse:
    """Represents the end-to-end output of a RAG query execution."""

    question: str
    answer: str
    retrieved_chunks: list[SearchResult]
    referenced_inquiries: list[str]  # Normalized document identifiers (e.g. ['IP_001_2023'])
    latency_ms: float
    pipeline_name: str
    cited_inquiries: list[str] = field(default_factory=list)  # Document identifiers explicitly cited by generator
    raw_answer: str | None = None  # Raw unparsed generator completion
    metadata: dict[str, Any] = field(default_factory=dict)
    prompt: str | None = None
    system_prompt: str | None = None
    pre_rerank_chunks: list[SearchResult] = field(default_factory=list)


@runtime_checkable
class RAGPipeline(Protocol):
    """Protocol for end-to-end retrieval-augmented generation pipelines."""

    @property
    def name(self) -> str:
        """Name of the pipeline architecture."""
        ...

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        """Execute the end-to-end RAG pipeline for a given question."""
        ...
