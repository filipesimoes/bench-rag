"""Unit tests for individual question Markdown reporting."""

from pathlib import Path

from bench_rag.core.chunker import Chunk
from bench_rag.core.evaluator import (
    EvaluationSummary,
    GroundTruthItem,
    LexicalMetrics,
    LLMJudgeMetrics,
    RetrievalMetrics,
    SampleEvaluation,
    TaxonomyAggregate,
)
from bench_rag.core.pipeline import RAGResponse
from bench_rag.core.vector_store import SearchResult
from bench_rag.reporting.markdown_reporter import (
    generate_all_question_reports,
    generate_markdown_report,
    generate_question_report,
)


def _create_dummy_sample(with_rerank: bool = True) -> SampleEvaluation:
    item = GroundTruthItem(
        id="GT-T1-C1",
        typology="T1",
        complexity="C1",
        question="Who is the nominee director?",
        reference_answer="The nominee director is Carlos Mendes.",
        inquiry_references=["IP_001_2023_FRAUD"],
    )

    chunk1 = Chunk(
        id="chunk_1",
        document_id="IP_001_2023_FRAUD",
        text="Carlos Mendes operates as the strawman nominee director for Nexus Global Holdings.",
        metadata={"page_number": 2},
    )
    chunk2 = Chunk(
        id="chunk_2",
        document_id="IP_002_2023_LAUNDERING",
        text="Orion Logistics conducted structured deposits under reporting limits.",
        metadata={"page_number": 5},
    )
    chunk3 = Chunk(
        id="chunk_3",
        document_id="IP_003_2023_CONSPIRACY",
        text="Banque de Genève wire transfers routed to shipyards.",
        metadata={"page_number": 1},
    )

    if with_rerank:
        # Pre-rerank pool: chunk2 was #1, chunk1 was #2, chunk3 was #3
        pre_rerank = [
            SearchResult(chunk=chunk2, score=0.88),
            SearchResult(chunk=chunk1, score=0.82),
            SearchResult(chunk=chunk3, score=0.65),
        ]
        # After rerank: chunk1 promoted to #1, chunk2 demoted to #2, chunk3 dropped
        post_chunk1 = Chunk(
            id="chunk_1",
            document_id="IP_001_2023_FRAUD",
            text=chunk1.text,
            metadata={
                "page_number": 2,
                "initial_retrieval_rank": 2,
                "initial_retrieval_score": 0.82,
            },
        )
        post_chunk2 = Chunk(
            id="chunk_2",
            document_id="IP_002_2023_LAUNDERING",
            text=chunk2.text,
            metadata={
                "page_number": 5,
                "initial_retrieval_rank": 1,
                "initial_retrieval_score": 0.88,
            },
        )
        retrieved = [
            SearchResult(chunk=post_chunk1, score=0.96),
            SearchResult(chunk=post_chunk2, score=0.74),
        ]
        reranker_name = "MockReranker"
    else:
        pre_rerank = []
        retrieved = [
            SearchResult(chunk=chunk1, score=0.85),
            SearchResult(chunk=chunk2, score=0.80),
        ]
        reranker_name = None

    response = RAGResponse(
        question=item.question,
        answer="The nominee director is Carlos Mendes. [Sources: IP_001_2023_FRAUD]",
        retrieved_chunks=retrieved,
        referenced_inquiries=["IP_001_2023_FRAUD"],
        latency_ms=125.4,
        pipeline_name="test_pipeline",
        metadata={
            "model_name": "mock-gen",
            "top_k": 2,
            "reranker": reranker_name,
        },
        prompt="Context Evidence:\n[Document: IP_001_2023_FRAUD]...\n\nQuestion: Who is the nominee director?\n\nAnswer:",
        system_prompt="You are an expert forensic intelligence analyst examining criminal police inquiry reports.",
        pre_rerank_chunks=pre_rerank,
    )

    retrieval = RetrievalMetrics(
        precision_at_k=1.0,
        recall_at_k=1.0,
        hit_rate=1.0,
        mrr=1.0,
    )
    lexical = LexicalMetrics(
        rouge_l=0.92,
        token_f1=0.95,
        bleu_4=0.85,
    )
    judge = LLMJudgeMetrics(
        faithfulness=0.98,
        answer_relevance=0.95,
        context_recall=0.90,
        critique="The generated answer directly addresses the query using verified forensic evidence.",
        prompt="Question:\nWho is the nominee director?...\nProvide your JSON evaluation:",
        system_prompt="You are a rigorous, impartial forensic intelligence auditor.",
    )

    return SampleEvaluation(
        item=item,
        response=response,
        retrieval=retrieval,
        lexical=lexical,
        llm_judge=judge,
    )


def test_generate_question_report_with_rerank(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=True)
    out_file = tmp_path / "GT-T1-C1.md"

    content = generate_question_report(sample, output_path=out_file)
    assert out_file.exists()

    # 1. Header & Metadata
    assert "# Benchmark Sample Report: `GT-T1-C1`" in content
    assert "**Pipeline Architecture**: `test_pipeline`" in content
    assert "**Investigative Typology**: `T1`" in content
    assert "**Complexity Level**: `C1`" in content
    assert "**Reranker Engine**: `MockReranker`" in content
    assert "125.4 ms" in content

    # 2. Scorecard
    assert "| Retrieval | Precision@K | 1.0000 |" in content
    assert "| LLM-as-a-Judge | Faithfulness (Groundedness) | 0.9800 |" in content
    assert "The generated answer directly addresses the query" in content

    # 3. Answers
    assert "Who is the nominee director?" in content
    assert "Carlos Mendes. [Sources: IP_001_2023_FRAUD]" in content
    assert "The nominee director is Carlos Mendes." in content

    # 4. Prompts
    assert "You are an expert forensic intelligence analyst" in content
    assert "Context Evidence:\n[Document: IP_001_2023_FRAUD]" in content
    assert "You are a rigorous, impartial forensic intelligence auditor." in content
    assert "Provide your JSON evaluation:" in content

    # 5. Reranking transition and chunks
    assert "5.1 Reranking Transition & Candidate Pool Overview" in content
    assert (
        "| #1 | #2 | +1 | `IP_001_2023_FRAUD` | `chunk_1` | 2 | 0.8200 | 0.9600 | **Promoted to Context** |"
        in content
    )
    assert (
        "| *Dropped* | #3 | - | `IP_003_2023_CONSPIRACY` | `chunk_3` | 1 | 0.6500 | - | *Dropped from Context* |"
        in content
    )

    # Promoted chunk text
    assert "Carlos Mendes operates as the strawman" in content

    # Dropped chunk in <details>
    assert "<details>" in content
    assert "<summary><strong>Candidate #3 (Dropped)</strong>" in content
    assert "Banque de Genève wire transfers" in content


def test_generate_question_report_without_rerank(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=False)
    out_file = tmp_path / "GT-T1-C1.md"

    content = generate_question_report(sample, output_path=out_file)
    assert out_file.exists()

    assert "**Reranker Engine**: `None (Direct Retrieval)`" in content
    assert "Reranker was not configured for this pipeline." in content
    assert "5.1 Retrieved Context Chunks" in content
    assert "| #1 | `IP_001_2023_FRAUD` | `chunk_1` | 2 | 0.8500 |" in content
    assert "Carlos Mendes operates as the strawman" in content
    assert "<details>" not in content


def test_generate_all_question_reports(tmp_path: Path):
    s1 = _create_dummy_sample(with_rerank=True)
    s2 = _create_dummy_sample(with_rerank=False)
    s2.item.id = "GT-T2-C1"

    dummy_agg = TaxonomyAggregate(
        sample_count=2,
        mean_precision_at_k=1.0,
        mean_recall_at_k=1.0,
        mean_hit_rate=1.0,
        mean_mrr=1.0,
        mean_rouge_l=0.9,
        mean_token_f1=0.9,
        mean_bleu_4=0.8,
        mean_faithfulness=0.95,
        mean_answer_relevance=0.95,
        mean_context_recall=0.9,
        mean_latency_ms=120.0,
    )
    summary = EvaluationSummary(
        pipeline_name="test_pipeline",
        total_samples=2,
        overall=dummy_agg,
        by_typology={"T1": dummy_agg, "T2": dummy_agg},
        by_complexity={"C1": dummy_agg},
        matrix={"T1": {"C1": dummy_agg}, "T2": {"C1": dummy_agg}},
        samples=[s1, s2],
    )

    q_dir = tmp_path / "questions"
    generated = generate_all_question_reports(summary, output_dir=q_dir)

    assert len(generated) == 2
    assert (q_dir / "GT-T1-C1.md").exists()
    assert (q_dir / "GT-T2-C1.md").exists()


def test_main_markdown_report_includes_question_links(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=True)
    dummy_agg = TaxonomyAggregate(
        sample_count=1,
        mean_precision_at_k=1.0,
        mean_recall_at_k=1.0,
        mean_hit_rate=1.0,
        mean_mrr=1.0,
        mean_rouge_l=0.9,
        mean_token_f1=0.9,
        mean_bleu_4=0.8,
        mean_faithfulness=0.95,
        mean_answer_relevance=0.95,
        mean_context_recall=0.9,
        mean_latency_ms=120.0,
    )
    summary = EvaluationSummary(
        pipeline_name="test_pipeline",
        total_samples=1,
        overall=dummy_agg,
        by_typology={"T1": dummy_agg},
        by_complexity={"C1": dummy_agg},
        matrix={"T1": {"C1": dummy_agg}},
        samples=[sample],
    )

    out_file = tmp_path / "evaluation_report.md"
    content = generate_markdown_report(summary, output_path=out_file, question_rel_dir="questions")

    assert out_file.exists()
    assert "[View Detailed Question Report](questions/GT-T1-C1.md)" in content

    # Test with custom subfolder for compare
    content_compare = generate_markdown_report(
        summary, question_rel_dir="questions/standard_dense_rag"
    )
    assert (
        "[View Detailed Question Report](questions/standard_dense_rag/GT-T1-C1.md)"
        in content_compare
    )
