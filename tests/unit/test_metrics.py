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


def test_retrieval_metrics_chunk_level_precision_and_mrr():
    # Two chunks from the same relevant doc at k=2 -> Precision@2 = 1.0, Recall@2 = 1.0
    retrieved_same_rel = ["IP_001_2023_FRAUD", "IP_001_2023_FRAUD"]
    gt_single = ["IP_001_2023_FRAUD"]
    res1 = compute_retrieval_metrics(retrieved_same_rel, gt_single, k=2)
    assert res1.precision_at_k == 1.0
    assert res1.recall_at_k == 1.0
    assert res1.mrr == 1.0

    # Two chunks from the same irrelevant doc followed by a relevant chunk at rank 3 -> MRR = 1/3
    retrieved_dup_irrel = ["OTHER_DOC", "OTHER_DOC", "IP_001_2023_FRAUD"]
    res2 = compute_retrieval_metrics(retrieved_dup_irrel, gt_single, k=3)
    assert res2.precision_at_k == 0.3333
    assert res2.recall_at_k == 1.0
    assert res2.mrr == 0.3333


def test_lexical_metrics_identical():
    text = "Carlos Mendes acted as strawman nominee director."
    res = compute_lexical_metrics(text, text)

    assert res.token_f1 == 1.0
    assert res.rouge_l == 1.0
    assert res.bleu_4 == 1.0


def test_lexical_metrics_short_identical_bleu():
    for short_text in ["Sim", "IP 001/2023", "Carlos Mendes diretor"]:
        res = compute_lexical_metrics(short_text, short_text)
        assert res.token_f1 == 1.0
        assert res.rouge_l == 1.0
        assert res.bleu_4 == pytest.approx(1.0)


def test_lexical_metrics_compound_numbers_and_symmetric_rouge():
    from bench_rag.evaluation.metrics.lexical import calculate_rouge_l, tokenize_text

    tokens = tokenize_text("Ref: 012/2023, valor US$ 10,000.50 em 15-08-2023.")
    assert "012/2023" in tokens
    assert "10,000.50" in tokens
    assert "15-08-2023" in tokens

    a = "O investigado transferiu 10,000.00 para a conta."
    b = "O investigado transferiu 20,000.00 para a conta."
    res = compute_lexical_metrics(a, b)
    assert res.token_f1 < 1.0

    # Symmetric ROUGE-L (beta = 1.0)
    short_s = "Carlos Mendes"
    long_s = "Carlos Mendes atuou como diretor nominal da empresa."
    assert calculate_rouge_l(short_s, long_s) == pytest.approx(calculate_rouge_l(long_s, short_s))


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
    assert 0.0 <= res.answer_correctness <= 1.0
    assert res.critique is not None


def test_llm_judge_clamp_and_json_extraction():
    from bench_rag.core.generator import GenerationResult, GeneratorLLM

    class CustomJSONGenerator(GeneratorLLM):
        def generate(self, prompt, **kwargs):
            return GenerationResult(
                answer=(
                    'Here is the evaluation (note {brackets} in preface):\n'
                    '{"faithfulness": 1.5, "answer_relevance": -0.2, '
                    '"context_recall": 0.75, "answer_correctness": 0.9, '
                    '"critique": "Clamped test"}'
                ),
                prompt=prompt,
                model_name="test-judge",
            )

    judge = LLMJudge(generator=CustomJSONGenerator())
    res = judge.evaluate(
        question="Q",
        answer="A",
        context="C",
        reference_answer="R",
    )
    assert res.faithfulness == 1.0
    assert res.answer_relevance == 0.0
    assert res.context_recall == 0.75
    assert res.answer_correctness == 0.9
    assert res.critique == "Clamped test"


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


def test_llm_judge_timeout_fail_fast():
    import httpx

    from bench_rag.core.generator import GeneratorLLM

    class TimingOutGenerator(GeneratorLLM):
        def generate(self, prompt, **kwargs):
            raise httpx.ReadTimeout("The read operation timed out during judge scoring")

    judge = LLMJudge(generator=TimingOutGenerator())
    with pytest.raises(httpx.ReadTimeout, match="timed out during judge scoring"):
        judge.evaluate(
            question="What is the role of TEST AGENT?",
            answer="TEST AGENT was the designated officer.",
            context="Synthetic record indicates TEST AGENT was the designated officer.",
            reference_answer="TEST AGENT was the designated officer.",
        )


def test_retrieval_metrics_fewer_chunks_than_k():
    """Precision@K divides by k even when fewer than k chunks are returned."""
    retrieved = ["IP_001_2023_FRAUD"]
    gt = ["IP_001_2023_FRAUD"]
    res = compute_retrieval_metrics(retrieved, gt, k=5)
    assert res.precision_at_k == 0.2
    assert res.recall_at_k == 1.0
    assert res.hit_rate == 1.0
    assert res.mrr == 1.0


def test_bleu_4_partial_overlap_uninflated():
    """Higher-order n-grams with overlap > 0 use exact precision without add-1 inflation."""
    from bench_rag.evaluation.metrics.lexical import calculate_bleu_4

    # 4 tokens each; unigrams: 3/4, bigrams: 2/3, trigrams: 1/2, 4-grams: 0 (smoothed to 1/(1+1)=0.5)
    cand = "alpha beta gamma delta"
    ref = "alpha beta gamma epsilon"
    expected_p = [3 / 4, 2 / 3, 1 / 2, 1 / 2]
    import math

    expected_bleu = round(math.exp(sum(math.log(p) for p in expected_p) / 4), 4)
    assert calculate_bleu_4(cand, ref) == expected_bleu


def test_llm_judge_missing_answer_correctness_raises():
    """LLMJudge.evaluate must require answer_correctness rather than copying context_recall."""
    from bench_rag.core.generator import GenerationResult, GeneratorLLM

    class IncompleteJSONGenerator(GeneratorLLM):
        def generate(self, prompt, **kwargs):
            return GenerationResult(
                answer='{"faithfulness": 0.9, "answer_relevance": 0.9, "context_recall": 0.8}',
                prompt=prompt,
                model_name="test-judge",
            )

    judge = LLMJudge(generator=IncompleteJSONGenerator())
    with pytest.raises(KeyError, match="answer_correctness"):
        judge.evaluate(
            question="What is the role of TEST AGENT?",
            answer="TEST AGENT was the designated officer.",
            context="Synthetic record indicates TEST AGENT was the designated officer.",
            reference_answer="TEST AGENT was the designated officer.",
        )

