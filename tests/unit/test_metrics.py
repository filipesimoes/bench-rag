import pytest

from bench_rag.evaluation.metrics.lexical import (
    compute_lexical_metrics,
)
from bench_rag.evaluation.metrics.llm_judge import LLMJudge
from bench_rag.evaluation.metrics.retrieval import compute_retrieval_metrics
from bench_rag.generators.mock_generator import MockGeneratorLLM


def test_retrieval_metrics_perfect_match():
    retrieved = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]
    gt = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]

    res = compute_retrieval_metrics(retrieved, gt, k=2)
    assert res.precision_at_k == 1.0
    assert res.recall_at_k == 1.0
    assert res.hit_rate == 1.0
    assert res.mrr == 1.0


def test_retrieval_metrics_partial_match():
    retrieved = ["OTHER_DOC", "IP_001_2023_FRAUD"]
    gt = ["IP_001_2023_FRAUD", "IP_002_2023_LAUNDERING"]

    res = compute_retrieval_metrics(retrieved, gt, k=2)
    assert res.precision_at_k == 0.5
    assert res.recall_at_k == 0.5
    assert res.hit_rate == 1.0
    assert res.mrr == 0.5  # First hit is at rank 2 -> 1/2 = 0.5


def test_retrieval_metrics_no_match():
    retrieved = ["OTHER_A", "OTHER_B"]
    gt = ["IP_001_2023_FRAUD"]

    res = compute_retrieval_metrics(retrieved, gt, k=2)
    assert res.precision_at_k == 0.0
    assert res.recall_at_k == 0.0
    assert res.hit_rate == 0.0
    assert res.mrr == 0.0


def test_lexical_metrics_identical():
    text = "Carlos Mendes acted as strawman nominee director."
    res = compute_lexical_metrics(text, text)

    assert res.token_f1 == 1.0
    assert res.rouge_l == 1.0
    assert res.bleu_4 == 1.0


def test_lexical_metrics_disjoint():
    cand = "Apple banana orange."
    ref = "Zebras elephants giraffes."
    res = compute_lexical_metrics(cand, ref)

    assert res.token_f1 == 0.0
    assert res.rouge_l == 0.0
    assert res.bleu_4 == 0.0


def test_lexical_metrics_partial():
    cand = "Carlos Mendes is the nominee director."
    ref = "Carlos Mendes acted as the nominee director with low salary."
    res = compute_lexical_metrics(cand, ref)

    assert 0.0 < res.token_f1 < 1.0
    assert 0.0 < res.rouge_l < 1.0


def test_llm_judge_mock():
    judge = LLMJudge(generator=MockGeneratorLLM())
    res = judge.evaluate(
        question="Who is the nominee director?",
        answer="Carlos Mendes is the nominee director.",
        context="Nexus records identify Carlos Mendes as nominee director.",
        reference_answer="Carlos Mendes is the nominee director.",
    )

    assert 0.0 <= res.faithfulness <= 1.0
    assert 0.0 <= res.answer_relevance <= 1.0
    assert 0.0 <= res.context_recall <= 1.0
    assert res.critique is not None


def test_get_judge_factory():
    from bench_rag.evaluation.metrics.llm_judge import get_judge
    from bench_rag.generators.openai_generator import OpenAIGeneratorLLM

    j1 = get_judge("mock")
    assert isinstance(j1, LLMJudge)

    j2 = get_judge(
        "openai",
        model_name="judge-llm",
        base_url="http://custom-judge:8002/v1",
        api_key="judge-key",
    )
    assert isinstance(j2, LLMJudge)
    assert isinstance(j2.generator, OpenAIGeneratorLLM)
    assert j2.generator.model == "judge-llm"
    assert j2.generator.base_url == "http://custom-judge:8002/v1"
    assert j2.generator.api_key == "judge-key"

    with pytest.raises(ValueError, match="Unsupported judge type"):
        get_judge("invalid")
