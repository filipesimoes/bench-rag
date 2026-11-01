"""Core EvaluationEngine protocol and evaluation data models."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from bench_rag.core.pipeline import RAGPipeline, RAGResponse


@dataclass
class GroundTruthItem:
    """Represents a benchmark row from the ground truth CSV."""

    id: str
    typology: str  # T1, T2, T3
    complexity: str  # C1, C2, C3
    question: str
    reference_answer: str
    inquiry_references: list[str]  # Normalized document identifiers, e.g. ["IP_001_2023"]


@dataclass
class RetrievalMetrics:
    """Retrieval evaluation metrics against ground-truth inquiry references."""

    precision_at_k: float
    recall_at_k: float
    hit_rate: float
    mrr: float


@dataclass
class LexicalMetrics:
    """Lexical comparison metrics against ground-truth reference answer."""

    rouge_l: float
    token_f1: float
    bleu_4: float


@dataclass
class LLMJudgeMetrics:
    """LLM-as-a-judge quality metrics."""

    faithfulness: float  # Grounding in retrieved context (0.0 - 1.0)
    answer_relevance: float  # Alignment with the query intent (0.0 - 1.0)
    context_recall: float  # Grounding against reference answer (0.0 - 1.0)
    critique: str | None = None
    prompt: str | None = None
    system_prompt: str | None = None


@dataclass
class CitationMetrics:
    """Generation citation evaluation metrics."""

    precision: float
    recall: float
    hallucination_rate: float
    cited_count: int


@dataclass
class SampleEvaluation:
    """Evaluation output for a single benchmark query."""

    item: GroundTruthItem
    response: RAGResponse
    retrieval: RetrievalMetrics
    lexical: LexicalMetrics
    llm_judge: LLMJudgeMetrics
    citation: CitationMetrics | None = None


@dataclass
class TaxonomyAggregate:
    """Aggregated metrics across a subset of benchmark queries."""

    sample_count: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_hit_rate: float
    mean_mrr: float
    mean_rouge_l: float
    mean_token_f1: float
    mean_bleu_4: float
    mean_faithfulness: float
    mean_answer_relevance: float
    mean_context_recall: float
    mean_latency_ms: float
    mean_citation_precision: float = 0.0
    mean_citation_recall: float = 0.0
    mean_citation_hallucination_rate: float = 0.0


@dataclass
class EvaluationSummary:
    """Complete summary of an evaluation run with taxonomy aggregations."""

    pipeline_name: str
    total_samples: int
    overall: TaxonomyAggregate
    by_typology: dict[str, TaxonomyAggregate]  # T1, T2, T3
    by_complexity: dict[str, TaxonomyAggregate]  # C1, C2, C3
    matrix: dict[str, dict[str, TaxonomyAggregate]]  # Typology x Complexity
    samples: list[SampleEvaluation] = field(default_factory=list)


@runtime_checkable
class EvaluationProgressCallback(Protocol):
    """Protocol for receiving real-time benchmark evaluation progress updates."""

    def __call__(
        self,
        current: int,
        total: int,
        item: GroundTruthItem,
        stage: str = "evaluating",
    ) -> None:
        """Invoked when evaluation progresses.

        Args:
            current: 1-based index of current query.
            total: Total number of queries in the dataset.
            item: GroundTruthItem currently being evaluated.
            stage: Active stage ('rag_query', 'llm_judge', 'completed').
        """
        ...


@runtime_checkable
class EvaluationEngine(Protocol):
    """Protocol for evaluating a RAG pipeline against a ground-truth benchmark."""

    def evaluate(
        self,
        dataset: list[GroundTruthItem],
        pipeline: RAGPipeline,
        top_k: int | None = None,
        progress_callback: EvaluationProgressCallback | None = None,
    ) -> EvaluationSummary:
        """Run inference and compute all metrics aggregated across taxonomies."""
        ...
