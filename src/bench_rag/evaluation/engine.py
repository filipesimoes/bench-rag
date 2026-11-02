"""Evaluation engine executing benchmark questions and computing aggregated metrics."""

import csv
import logging
from pathlib import Path
from typing import Any

from bench_rag.core.evaluator import (
    EvaluationEngine,
    EvaluationProgressCallback,
    EvaluationSummary,
    GroundTruthItem,
    SampleEvaluation,
)
from bench_rag.core.pipeline import RAGPipeline
from bench_rag.evaluation.metadata import extract_execution_metadata
from bench_rag.evaluation.metrics.citation import compute_citation_metrics
from bench_rag.evaluation.metrics.lexical import compute_lexical_metrics
from bench_rag.evaluation.metrics.llm_judge import LLMJudge, get_judge
from bench_rag.evaluation.metrics.retrieval import compute_retrieval_metrics
from bench_rag.evaluation.taxonomy import build_evaluation_summary

logger = logging.getLogger(__name__)


def load_ground_truth_csv(csv_path: Path | str) -> list[GroundTruthItem]:
    """Load benchmark ground truth dataset from CSV.

    The CSV must contain:
    - id: Unique query identifier
    - typology: T1, T2, T3
    - complexity: C1, C2, C3
    - question: Investigative query
    - reference_answer: Ground truth reference answer
    - inquiry_references: Semicolon-separated inquiry identifiers (e.g. IP_001_2023_FRAUD; IP_002_2023_LAUNDERING)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {path}")

    items: list[GroundTruthItem] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {
            "id",
            "typology",
            "complexity",
            "question",
            "reference_answer",
            "inquiry_references",
        }
        if not required_cols.issubset(set(reader.fieldnames or [])):
            missing = required_cols - set(reader.fieldnames or [])
            raise ValueError(f"Ground truth CSV missing required columns: {missing}")

        for row in reader:
            raw_refs = row["inquiry_references"]
            # Semicolon-separated references per user specification
            parsed_refs = [ref.strip() for ref in raw_refs.split(";") if ref.strip()]

            items.append(
                GroundTruthItem(
                    id=row["id"].strip(),
                    typology=row["typology"].strip().upper(),
                    complexity=row["complexity"].strip().upper(),
                    question=row["question"].strip(),
                    reference_answer=row["reference_answer"].strip(),
                    inquiry_references=parsed_refs,
                )
            )

    return items


class StandardEvaluationEngine(EvaluationEngine):
    """Executes benchmark evaluation against a ground-truth dataset."""

    def __init__(
        self,
        judge: LLMJudge | None = None,
        judge_type: str = "mock",
        judge_model: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        judge_enable_thinking: bool | None = None,
        judge_dialect: str | None = None,
        judge_max_tokens: int | None = None,
        judge_timeout: float | None = None,
        judge_max_retries: int | None = None,
    ):
        if judge is not None:
            self.judge = judge
        else:
            self.judge = get_judge(
                judge_type=judge_type,
                model_name=judge_model,
                base_url=judge_base_url,
                api_key=judge_api_key,
                enable_thinking=judge_enable_thinking,
                dialect=judge_dialect,
                max_tokens=judge_max_tokens,
                timeout=judge_timeout,
                max_retries=judge_max_retries,
            )

    def evaluate(
        self,
        dataset: list[GroundTruthItem],
        pipeline: RAGPipeline,
        top_k: int | None = None,
        progress_callback: EvaluationProgressCallback | None = None,
        execution_metadata: dict[str, Any] | None = None,
    ) -> EvaluationSummary:
        """Run inference and evaluate all metrics across the dataset."""
        sample_evaluations: list[SampleEvaluation] = []
        total = len(dataset)
        effective_k = top_k if top_k is not None else getattr(pipeline, "top_k", 5)

        for idx, item in enumerate(dataset, start=1):
            if progress_callback:
                progress_callback(idx, total, item, "rag_query")

            # 1. Execute RAG pipeline (fail-fast on unhandled pipeline errors)
            response = pipeline.query(item.question, top_k=effective_k)

            # 2. Retrieval metrics against ground truth references (chunk-level rank & precision)
            chunk_doc_ids = [
                res.chunk.document_id
                for res in response.retrieved_chunks
                if res.chunk.document_id
            ] or response.referenced_inquiries
            retrieval_res = compute_retrieval_metrics(
                retrieved_inquiries=chunk_doc_ids,
                ground_truth_references=item.inquiry_references,
                k=effective_k,
            )

            # 3. Citation metrics against ground truth references (generator attribution)
            citation_res = compute_citation_metrics(
                cited_inquiries=response.cited_inquiries,
                ground_truth_references=item.inquiry_references,
                retrieved_inquiries=response.referenced_inquiries,
            )

            # 4. Lexical metrics against reference answer
            lexical_res = compute_lexical_metrics(
                candidate=response.answer,
                reference=item.reference_answer,
            )

            # 5. Mandatory LLM-as-a-judge quality metrics
            if progress_callback:
                progress_callback(idx, total, item, "llm_judge")

            context_blocks = []
            for res in response.retrieved_chunks:
                doc_id = res.chunk.document_id
                if doc_id:
                    page = res.chunk.metadata.get("page_number", "N/A")
                    context_blocks.append(f"[Document: {doc_id}] (Page {page}):\n{res.chunk.text}")
                else:
                    context_blocks.append(res.chunk.text)
            context_text = "\n\n".join(context_blocks)
            judge_res = self.judge.evaluate(
                question=item.question,
                answer=response.answer,
                context=context_text,
                reference_answer=item.reference_answer,
            )

            sample_evaluations.append(
                SampleEvaluation(
                    item=item,
                    response=response,
                    retrieval=retrieval_res,
                    lexical=lexical_res,
                    llm_judge=judge_res,
                    citation=citation_res,
                )
            )

            if progress_callback:
                progress_callback(idx, total, item, "completed")

        # 5. Aggregate across taxonomies
        meta = execution_metadata or extract_execution_metadata(
            pipeline=pipeline,
            judge=self.judge,
            top_k=effective_k,
            total_samples=total,
        )
        return build_evaluation_summary(
            pipeline_name=pipeline.name,
            samples=sample_evaluations,
            execution_metadata=meta,
        )
