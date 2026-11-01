"""Unit tests for taxonomy aggregations (T1-T3 x C1-C3)."""

from bench_rag.core.evaluator import (
    GroundTruthItem,
    LexicalMetrics,
    LLMJudgeMetrics,
    RetrievalMetrics,
    SampleEvaluation,
)
from bench_rag.core.pipeline import RAGResponse
from bench_rag.evaluation.taxonomy import build_evaluation_summary


def test_taxonomy_aggregation_matrix():
    samples = [
        SampleEvaluation(
            item=GroundTruthItem("Q1", "T1", "C1", "q1", "a1", ["doc1"]),
            response=RAGResponse("q1", "a1", [], ["doc1"], 10.0, "test_rag"),
            retrieval=RetrievalMetrics(1.0, 1.0, 1.0, 1.0),
            lexical=LexicalMetrics(0.8, 0.8, 0.7),
            llm_judge=LLMJudgeMetrics(0.9, 0.9, 0.9),
        ),
        SampleEvaluation(
            item=GroundTruthItem("Q2", "T1", "C2", "q2", "a2", ["doc1", "doc2"]),
            response=RAGResponse("q2", "a2", [], ["doc1"], 15.0, "test_rag"),
            retrieval=RetrievalMetrics(0.5, 0.5, 1.0, 1.0),
            lexical=LexicalMetrics(0.6, 0.6, 0.5),
            llm_judge=LLMJudgeMetrics(0.8, 0.8, 0.8),
        ),
        SampleEvaluation(
            item=GroundTruthItem("Q3", "T2", "C1", "q3", "a3", ["doc3"]),
            response=RAGResponse("q3", "a3", [], ["doc3"], 20.0, "test_rag"),
            retrieval=RetrievalMetrics(1.0, 1.0, 1.0, 1.0),
            lexical=LexicalMetrics(0.9, 0.9, 0.8),
            llm_judge=LLMJudgeMetrics(1.0, 1.0, 1.0),
        ),
    ]

    summary = build_evaluation_summary("test_rag", samples)

    assert summary.total_samples == 3
    assert "T1" in summary.by_typology
    assert "T2" in summary.by_typology
    assert summary.by_typology["T1"].sample_count == 2
    assert summary.by_typology["T2"].sample_count == 1

    assert "C1" in summary.by_complexity
    assert "C2" in summary.by_complexity
    assert summary.by_complexity["C1"].sample_count == 2

    # Check 2D matrix
    assert "T1" in summary.matrix
    assert "C1" in summary.matrix["T1"]
    assert "C2" in summary.matrix["T1"]
    assert summary.matrix["T1"]["C1"].mean_recall_at_k == 1.0
    assert summary.matrix["T1"]["C2"].mean_recall_at_k == 0.5
