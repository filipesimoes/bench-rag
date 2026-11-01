"""Unit tests for CitationMetrics calculation."""

from bench_rag.evaluation.metrics.citation import compute_citation_metrics


def test_citation_metrics_perfect_match():
    cited = ["IP_001_2023_FRAUD"]
    gt = ["IP_001_2023_FRAUD"]
    retrieved = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]

    res = compute_citation_metrics(cited, gt, retrieved)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.hallucination_rate == 0.0
    assert res.cited_count == 1


def test_citation_metrics_partial_recall():
    cited = ["IP_001_2023_FRAUD"]
    gt = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]
    retrieved = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]

    res = compute_citation_metrics(cited, gt, retrieved)
    assert res.precision == 1.0
    assert res.recall == 0.5
    assert res.hallucination_rate == 0.0
    assert res.cited_count == 1


def test_citation_metrics_with_hallucination():
    cited = ["IP_001_2023_FRAUD", "IP_999_HALLUCINATED"]
    gt = ["IP_001_2023_FRAUD"]
    retrieved = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]

    res = compute_citation_metrics(cited, gt, retrieved)
    assert res.precision == 0.5  # 1 match out of 2 cited
    assert res.recall == 1.0  # 1 match out of 1 gt
    assert res.hallucination_rate == 0.5  # IP_999 was not in retrieved context
    assert res.cited_count == 2


def test_citation_metrics_empty_cited():
    cited = []
    gt = ["IP_001_2023_FRAUD"]
    retrieved = ["IP_001_2023_FRAUD"]

    res = compute_citation_metrics(cited, gt, retrieved)
    assert res.precision == 0.0
    assert res.recall == 0.0
    assert res.hallucination_rate == 0.0
    assert res.cited_count == 0


def test_citation_metrics_case_insensitivity():
    cited = ["ip_001_2023_fraud"]
    gt = ["IP_001_2023_FRAUD"]
    retrieved = ["IP_001_2023_FRAUD"]

    res = compute_citation_metrics(cited, gt, retrieved)
    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.hallucination_rate == 0.0
