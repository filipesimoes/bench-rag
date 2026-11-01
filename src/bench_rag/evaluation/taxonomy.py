"""Taxonomy aggregation across investigative typologies (T1-T3) and complexities (C1-C3)."""

from collections import defaultdict

from bench_rag.core.evaluator import (
    EvaluationSummary,
    SampleEvaluation,
    TaxonomyAggregate,
)


def aggregate_samples(samples: list[SampleEvaluation]) -> TaxonomyAggregate:
    """Aggregate a list of SampleEvaluation instances into a TaxonomyAggregate."""
    n = len(samples)
    if n == 0:
        return TaxonomyAggregate(
            sample_count=0,
            mean_precision_at_k=0.0,
            mean_recall_at_k=0.0,
            mean_hit_rate=0.0,
            mean_mrr=0.0,
            mean_rouge_l=0.0,
            mean_token_f1=0.0,
            mean_bleu_4=0.0,
            mean_faithfulness=0.0,
            mean_answer_relevance=0.0,
            mean_context_recall=0.0,
            mean_latency_ms=0.0,
        )

    return TaxonomyAggregate(
        sample_count=n,
        mean_precision_at_k=round(sum(s.retrieval.precision_at_k for s in samples) / n, 4),
        mean_recall_at_k=round(sum(s.retrieval.recall_at_k for s in samples) / n, 4),
        mean_hit_rate=round(sum(s.retrieval.hit_rate for s in samples) / n, 4),
        mean_mrr=round(sum(s.retrieval.mrr for s in samples) / n, 4),
        mean_rouge_l=round(sum(s.lexical.rouge_l for s in samples) / n, 4),
        mean_token_f1=round(sum(s.lexical.token_f1 for s in samples) / n, 4),
        mean_bleu_4=round(sum(s.lexical.bleu_4 for s in samples) / n, 4),
        mean_faithfulness=round(sum(s.llm_judge.faithfulness for s in samples) / n, 4),
        mean_answer_relevance=round(sum(s.llm_judge.answer_relevance for s in samples) / n, 4),
        mean_context_recall=round(sum(s.llm_judge.context_recall for s in samples) / n, 4),
        mean_latency_ms=round(sum(s.response.latency_ms for s in samples) / n, 2),
    )


def build_evaluation_summary(
    pipeline_name: str,
    samples: list[SampleEvaluation],
) -> EvaluationSummary:
    """Group sample evaluations by Typology, Complexity, and their 2D Matrix."""
    overall = aggregate_samples(samples)

    # Group by Typology (T1, T2, T3)
    typology_groups: dict[str, list[SampleEvaluation]] = defaultdict(list)
    for s in samples:
        typology_groups[s.item.typology.upper()].append(s)

    by_typology = {typ: aggregate_samples(group) for typ, group in sorted(typology_groups.items())}

    # Group by Complexity (C1, C2, C3)
    complexity_groups: dict[str, list[SampleEvaluation]] = defaultdict(list)
    for s in samples:
        complexity_groups[s.item.complexity.upper()].append(s)

    by_complexity = {
        comp: aggregate_samples(group) for comp, group in sorted(complexity_groups.items())
    }

    # Group by 2D Matrix (Typology x Complexity)
    matrix_groups: dict[str, dict[str, list[SampleEvaluation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for s in samples:
        t = s.item.typology.upper()
        c = s.item.complexity.upper()
        matrix_groups[t][c].append(s)

    matrix: dict[str, dict[str, TaxonomyAggregate]] = {}
    for t in sorted(matrix_groups.keys()):
        matrix[t] = {}
        for c in sorted(matrix_groups[t].keys()):
            matrix[t][c] = aggregate_samples(matrix_groups[t][c])

    return EvaluationSummary(
        pipeline_name=pipeline_name,
        total_samples=len(samples),
        overall=overall,
        by_typology=by_typology,
        by_complexity=by_complexity,
        matrix=matrix,
        samples=samples,
    )
