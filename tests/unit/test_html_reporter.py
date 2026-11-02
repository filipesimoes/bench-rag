"""Unit tests for HTML reporting module."""

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
from bench_rag.evaluation.metrics.citation import CitationMetrics
from bench_rag.reporting.html_reporter import (
    generate_all_question_reports,
    generate_comparison_report,
    generate_html_report,
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
        pre_rerank = [
            SearchResult(chunk=chunk2, score=0.88),
            SearchResult(chunk=chunk1, score=0.82),
            SearchResult(chunk=chunk3, score=0.65),
        ]
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
        answer="The nominee director is Carlos Mendes.",
        retrieved_chunks=retrieved,
        referenced_inquiries=["IP_001_2023_FRAUD"],
        cited_inquiries=["IP_001_2023_FRAUD"],
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
    citation = CitationMetrics(
        precision=1.0,
        recall=1.0,
        hallucination_rate=0.0,
        cited_count=1,
    )

    return SampleEvaluation(
        item=item,
        response=response,
        retrieval=retrieval,
        lexical=lexical,
        llm_judge=judge,
        citation=citation,
    )


def test_generate_question_report_with_rerank(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=True)
    out_file = tmp_path / "GT-T1-C1.html"

    content = generate_question_report(sample, output_path=out_file)
    assert out_file.exists()

    # 1. HTML structure and CSS
    assert "<!DOCTYPE html>" in content
    assert "<html lang=\"en\">" in content
    assert "<style>" in content
    assert "Back to Evaluation Report" in content

    # 2. Header & Metadata
    assert "Benchmark Sample Report: <code>GT-T1-C1</code>" in content
    assert "test_pipeline" in content
    assert "T1" in content
    assert "C1" in content
    assert "MockReranker" in content
    assert "125.4 ms" in content

    # 3. Scorecard
    assert "Precision@K" in content
    assert "1.0000" in content
    assert "0.9800" in content
    assert "The generated answer directly addresses the query" in content

    # 4. Answers
    assert "Who is the nominee director?" in content
    assert "The nominee director is Carlos Mendes." in content

    # 5. Prompts in <pre><code>
    assert "<pre><code>You are an expert forensic intelligence analyst" in content
    assert "Context Evidence:" in content

    # 6. Reranking transition and chunks
    assert "5.1 Reranking Transition &amp; Candidate Pool Overview" in content
    assert "Promoted to Context" in content
    assert "Dropped from Context" in content
    assert "Carlos Mendes operates as the strawman" in content

    # 7. Dropped candidate in <details><summary>
    assert "<details>" in content
    assert "<summary>" in content
    assert "Banque de Genève wire transfers" in content


def test_generate_question_report_without_rerank(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=False)
    out_file = tmp_path / "GT-T1-C1.html"

    content = generate_question_report(sample, output_path=out_file)
    assert out_file.exists()

    assert "None (Direct Retrieval)" in content
    assert "Reranker was not configured for this pipeline." in content
    assert "5.1 Retrieved Context Chunks" in content
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
    assert (q_dir / "GT-T1-C1.html").exists()
    assert (q_dir / "GT-T2-C1.html").exists()


def test_generate_html_report_overview(tmp_path: Path):
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

    out_file = tmp_path / "evaluation_report.html"
    content = generate_html_report(summary, output_path=out_file, question_rel_dir="questions")

    assert out_file.exists()
    assert "<!DOCTYPE html>" in content
    assert "RAG Benchmark Evaluation Report: <code>test_pipeline</code>" in content
    assert "Pipeline &amp; Models Configuration" in content
    assert "Generator Model" in content
    assert "Embedding Model" in content
    assert "Reranker Model" in content
    assert "LLM Judge Model" in content
    assert "Executive Summary" in content
    assert "Global Performance Metrics" in content
    assert "Typology Breakdown" in content
    assert "Complexity Breakdown" in content
    assert "Typology x Complexity Matrix" in content
    assert "<div class='card-title'>Context Recall</div>" in content
    assert "<th class='num'>Context Recall</th>" in content
    assert "Context Recall: <strong>0.9000</strong>" in content
    assert "href='questions/GT-T1-C1.html'" in content

    # Assert metric guide is present
    assert "<details class=\"metric-guide\">" in content
    assert "Evaluation Metrics Interpretation Guide" in content
    assert "Retrieval Metrics" in content
    assert "Citation Attribution Metrics" in content
    assert "Lexical Overlap Metrics" in content
    assert "LLM-as-a-Judge Forensic Metrics" in content
    assert "Citation Hallucination Rate" in content
    assert "Faithfulness (Groundedness)" in content

    # Test with custom question_rel_dir
    content_compare = generate_html_report(
        summary, question_rel_dir="questions/standard_dense_rag"
    )
    assert "href='questions/standard_dense_rag/GT-T1-C1.html'" in content_compare


def test_generate_comparison_report(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=True)
    dummy_agg_a = TaxonomyAggregate(
        sample_count=1,
        mean_precision_at_k=0.8,
        mean_recall_at_k=0.75,
        mean_hit_rate=1.0,
        mean_mrr=0.75,
        mean_rouge_l=0.8,
        mean_token_f1=0.8,
        mean_bleu_4=0.7,
        mean_faithfulness=0.9,
        mean_answer_relevance=0.9,
        mean_context_recall=0.85,
        mean_latency_ms=100.0,
    )
    dummy_agg_b = TaxonomyAggregate(
        sample_count=1,
        mean_precision_at_k=0.9,
        mean_recall_at_k=0.85,
        mean_hit_rate=1.0,
        mean_mrr=0.85,
        mean_rouge_l=0.85,
        mean_token_f1=0.88,
        mean_bleu_4=0.75,
        mean_faithfulness=0.95,
        mean_answer_relevance=0.95,
        mean_context_recall=0.90,
        mean_latency_ms=150.0,
    )
    summary_a = EvaluationSummary(
        pipeline_name="pipe_a",
        total_samples=1,
        overall=dummy_agg_a,
        by_typology={"T1": dummy_agg_a},
        by_complexity={"C1": dummy_agg_a},
        matrix={"T1": {"C1": dummy_agg_a}},
        samples=[sample],
    )
    summary_b = EvaluationSummary(
        pipeline_name="pipe_b",
        total_samples=1,
        overall=dummy_agg_b,
        by_typology={"T1": dummy_agg_b},
        by_complexity={"C1": dummy_agg_b},
        matrix={"T1": {"C1": dummy_agg_b}},
        samples=[sample],
    )

    out_file = tmp_path / "comparison_report.html"
    content = generate_comparison_report(
        summary_a,
        summary_b,
        output_path=out_file,
        report_a_href="pipe_a_report.html",
        report_b_href="pipe_b_report.html",
    )

    assert out_file.exists()
    assert "RAG Benchmark Architecture Comparison: <code>pipe_a</code> vs. <code>pipe_b</code>" in content
    assert "Pipeline Architectures &amp; Models Comparison" in content
    assert "href='pipe_a_report.html'" in content
    assert "href='pipe_b_report.html'" in content
    assert "+0.1000" in content  # Delta for recall or precision
    assert "badge-success" in content
    assert "<th class='num'>pipe_a CtxRec</th><th class='num'>pipe_b CtxRec</th>" in content
    assert "Typology Comparison (Recall@K / Faithfulness / Context Recall)" in content
    assert "<details class=\"metric-guide\">" in content
    assert "Evaluation Metrics Interpretation Guide" in content


def test_html_report_with_custom_execution_metadata(tmp_path: Path):
    sample = _create_dummy_sample(with_rerank=False)
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
        pipeline_name="custom_hybrid_rag",
        total_samples=1,
        overall=dummy_agg,
        by_typology={"T1": dummy_agg},
        by_complexity={"C1": dummy_agg},
        matrix={"T1": {"C1": dummy_agg}},
        samples=[sample],
        execution_metadata={
            "pipeline_name": "custom_hybrid_rag",
            "pipeline_type": "Hybrid BM25 + Dense RAG",
            "top_k": 8,
            "generator_model": "test-model-4o",
            "embedding_model": "test-embed-v2",
            "reranker_model": "test-bge-reranker",
            "judge_model": "test-judge-v1",
        },
    )

    out_file = tmp_path / "custom_report.html"
    content = generate_html_report(summary, output_path=out_file)

    assert "Hybrid BM25 + Dense RAG" in content
    assert "test-model-4o" in content
    assert "test-embed-v2" in content
    assert "test-bge-reranker" in content
    assert "test-judge-v1" in content
    assert "<code>8</code>" in content


def test_html_escaping():
    item = GroundTruthItem(
        id="GT-<test>&1",
        typology="T1",
        complexity="C1",
        question="Is <entity> & 'strawman' active?",
        reference_answer="Yes, <entity> is active & confirmed.",
        inquiry_references=["IP_<001>"],
    )
    response = RAGResponse(
        question=item.question,
        answer="Found <entity> in & records.",
        retrieved_chunks=[],
        referenced_inquiries=["IP_<001>"],
        latency_ms=50.0,
        pipeline_name="pipe_<name>",
    )
    sample = SampleEvaluation(
        item=item,
        response=response,
        retrieval=RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, hit_rate=1.0, mrr=1.0),
        lexical=LexicalMetrics(rouge_l=1.0, token_f1=1.0, bleu_4=1.0),
        llm_judge=LLMJudgeMetrics(
            faithfulness=1.0,
            answer_relevance=1.0,
            context_recall=1.0,
            critique="Critique with <brackets> & ampersand",
        ),
    )

    content = generate_question_report(sample)
    assert "&lt;entity&gt;" in content
    assert "&amp; &#x27;strawman&#x27;" in content or "&amp; 'strawman'" in content or "&amp;" in content
    assert "GT-&lt;test&gt;&amp;1" in content
    assert "Critique with &lt;brackets&gt; &amp; ampersand" in content
