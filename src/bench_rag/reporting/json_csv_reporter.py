"""Export evaluation results to JSON and CSV formats."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

from bench_rag.core.evaluator import EvaluationSummary


def export_evaluation_json(summary: EvaluationSummary, output_path: Path | str) -> None:
    """Export the entire EvaluationSummary to a formatted JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    data = asdict(summary)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_samples_csv(summary: EvaluationSummary, output_path: Path | str) -> None:
    """Export sample-level evaluation results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id",
        "typology",
        "complexity",
        "question",
        "reference_inquiries",
        "retrieved_inquiries",
        "cited_inquiries",
        "precision_at_k",
        "recall_at_k",
        "hit_rate",
        "mrr",
        "citation_precision",
        "citation_recall",
        "citation_hallucination_rate",
        "rouge_l",
        "token_f1",
        "bleu_4",
        "faithfulness",
        "answer_relevance",
        "context_recall",
        "latency_ms",
        "generated_answer",
        "reference_answer",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in summary.samples:
            cit = s.citation
            writer.writerow(
                {
                    "id": s.item.id,
                    "typology": s.item.typology,
                    "complexity": s.item.complexity,
                    "question": s.item.question,
                    "reference_inquiries": ";".join(s.item.inquiry_references),
                    "retrieved_inquiries": ";".join(s.response.referenced_inquiries),
                    "cited_inquiries": ";".join(s.response.cited_inquiries),
                    "precision_at_k": s.retrieval.precision_at_k,
                    "recall_at_k": s.retrieval.recall_at_k,
                    "hit_rate": s.retrieval.hit_rate,
                    "mrr": s.retrieval.mrr,
                    "citation_precision": cit.precision if cit else 0.0,
                    "citation_recall": cit.recall if cit else 0.0,
                    "citation_hallucination_rate": cit.hallucination_rate if cit else 0.0,
                    "rouge_l": s.lexical.rouge_l,
                    "token_f1": s.lexical.token_f1,
                    "bleu_4": s.lexical.bleu_4,
                    "faithfulness": s.llm_judge.faithfulness,
                    "answer_relevance": s.llm_judge.answer_relevance,
                    "context_recall": s.llm_judge.context_recall,
                    "latency_ms": s.response.latency_ms,
                    "generated_answer": s.response.answer,
                    "reference_answer": s.item.reference_answer,
                }
            )
