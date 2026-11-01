"""Evaluation subsystem exports."""

from bench_rag.evaluation.engine import StandardEvaluationEngine, load_ground_truth_csv
from bench_rag.evaluation.metrics.lexical import compute_lexical_metrics
from bench_rag.evaluation.metrics.llm_judge import LLMJudge, get_judge
from bench_rag.evaluation.metrics.retrieval import compute_retrieval_metrics
from bench_rag.evaluation.taxonomy import aggregate_samples, build_evaluation_summary

__all__ = [
    "LLMJudge",
    "StandardEvaluationEngine",
    "aggregate_samples",
    "build_evaluation_summary",
    "compute_lexical_metrics",
    "compute_retrieval_metrics",
    "get_judge",
    "load_ground_truth_csv",
]
