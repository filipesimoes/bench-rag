import re
from pathlib import Path

from bench_rag.core.evaluator import EvaluationSummary, SampleEvaluation, TaxonomyAggregate

TYPOLOGY_DESCRIPTIONS = {
    "T1": "T1 (Identification & Associations): Intermediaries, shell entities, strawmen (*laranjas*), corporate ownership trees, beneficial owners.",
    "T2": "T2 (Criminal Dynamics): Modus operandi, money laundering typologies, criminal jargon, asset concealment tactics.",
    "T3": "T3 (Tracking & Chronology): Asset flow trails, wire transfer sequences, chronological timeline of suspicious activities.",
}

COMPLEXITY_DESCRIPTIONS = {
    "C1": "C1 (Low / Factual): Single-document, literal fact extraction.",
    "C2": "C2 (Medium / Relational): Multi-document aggregation testing retrieval exhaustiveness.",
    "C3": "C3 (High / Analytical): Broad inferential context regarding criminal intent (*mens rea* / *dolo*).",
}


def _format_aggregate_table(title: str, aggregates: dict[str, TaxonomyAggregate]) -> str:
    """Format a markdown table from a dictionary of category -> TaxonomyAggregate."""
    lines = [
        f"### {title}",
        "",
        "| Category | Samples | Precision@K | Recall@K | Hit Rate | MRR | Cit. Prec. | Cit. Rec. | ROUGE-L | Token F1 | Faithfulness | Relevance | Context Recall | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cat, agg in aggregates.items():
        lines.append(
            f"| **{cat}** | {agg.sample_count} | {agg.mean_precision_at_k:.4f} | {agg.mean_recall_at_k:.4f} | "
            f"{agg.mean_hit_rate:.4f} | {agg.mean_mrr:.4f} | {agg.mean_citation_precision:.4f} | {agg.mean_citation_recall:.4f} | "
            f"{agg.mean_rouge_l:.4f} | {agg.mean_token_f1:.4f} | "
            f"{agg.mean_faithfulness:.4f} | {agg.mean_answer_relevance:.4f} | {agg.mean_context_recall:.4f} | "
            f"{agg.mean_latency_ms:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_markdown_report(
    summary: EvaluationSummary,
    output_path: Path | str | None = None,
    question_rel_dir: str | None = "questions",
) -> str:
    """Generate a comprehensive Markdown evaluation report."""
    ov = summary.overall
    lines = [
        f"# RAG Benchmark Evaluation Report: `{summary.pipeline_name}`",
        "",
        "## 1. Executive Summary",
        f"- **Pipeline Architecture**: `{summary.pipeline_name}`",
        f"- **Total Evaluated Benchmark Samples**: {summary.total_samples}",
        f"- **Mean Retrieval Recall@K**: {ov.mean_recall_at_k:.4f}",
        f"- **Mean Retrieval MRR**: {ov.mean_mrr:.4f}",
        f"- **Mean Citation Precision**: {ov.mean_citation_precision:.4f}",
        f"- **Mean Citation Recall**: {ov.mean_citation_recall:.4f}",
        f"- **Mean Answer Faithfulness (LLM Judge)**: {ov.mean_faithfulness:.4f}",
        f"- **Mean Answer Relevance (LLM Judge)**: {ov.mean_answer_relevance:.4f}",
        f"- **Mean Latency**: {ov.mean_latency_ms:.1f} ms",
        "",
        "## 2. Global Performance Metrics",
        "",
        "| Metric Category | Metric | Score |",
        "|---|---|---|",
        f"| Retrieval | Precision@K | {ov.mean_precision_at_k:.4f} |",
        f"| Retrieval | Recall@K | {ov.mean_recall_at_k:.4f} |",
        f"| Retrieval | Hit Rate@K | {ov.mean_hit_rate:.4f} |",
        f"| Retrieval | Mean Reciprocal Rank (MRR) | {ov.mean_mrr:.4f} |",
        f"| Citation Attribution | Citation Precision | {ov.mean_citation_precision:.4f} |",
        f"| Citation Attribution | Citation Recall | {ov.mean_citation_recall:.4f} |",
        f"| Citation Attribution | Citation Hallucination Rate | {ov.mean_citation_hallucination_rate:.4f} |",
        f"| Lexical Overlap | ROUGE-L | {ov.mean_rouge_l:.4f} |",
        f"| Lexical Overlap | Token F1 | {ov.mean_token_f1:.4f} |",
        f"| Lexical Overlap | BLEU-4 | {ov.mean_bleu_4:.4f} |",
        f"| LLM-as-a-Judge | Faithfulness (Groundedness) | {ov.mean_faithfulness:.4f} |",
        f"| LLM-as-a-Judge | Answer Relevance | {ov.mean_answer_relevance:.4f} |",
        f"| LLM-as-a-Judge | Context Recall | {ov.mean_context_recall:.4f} |",
        f"| Performance | Mean Latency | {ov.mean_latency_ms:.1f} ms |",
        "",
        "## 3. Performance by Investigative Typology",
        "- **T1 (Identification & Associations)**: Intermediaries/strawmen (*laranjas*), corporate ownership trees, beneficial owners.",
        "- **T2 (Criminal Dynamics)**: Modus operandi, money laundering typologies, criminal jargon, asset concealment tactics.",
        "- **T3 (Tracking & Chronology)**: Asset flow trails, wire transfer sequences, chronological timeline of suspicious activities.",
        "",
        _format_aggregate_table("Typology Breakdown", summary.by_typology),
        "## 4. Performance by Complexity Level",
        "- **C1 (Low / Factual)**: Single-document, literal fact extraction.",
        "- **C2 (Medium / Relational)**: Multi-document aggregation testing retrieval exhaustiveness.",
        "- **C3 (High / Analytical)**: Broad inferential context regarding criminal intent (*mens rea* / *dolo*).",
        "",
        _format_aggregate_table("Complexity Breakdown", summary.by_complexity),
        "## 5. Typology x Complexity Matrix",
        "",
        "| Typology | Complexity | Samples | Recall@K | MRR | Token F1 | Faithfulness | Relevance |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for typ, comp_dict in sorted(summary.matrix.items()):
        for comp, agg in sorted(comp_dict.items()):
            lines.append(
                f"| **{typ}** | **{comp}** | {agg.sample_count} | {agg.mean_recall_at_k:.4f} | "
                f"{agg.mean_mrr:.4f} | {agg.mean_token_f1:.4f} | {agg.mean_faithfulness:.4f} | "
                f"{agg.mean_answer_relevance:.4f} |"
            )

    lines.extend(
        [
            "",
            "## 6. Sample-Level Granular Results",
            "",
        ]
    )

    for s in summary.samples:
        safe_id = re.sub(r"[^\w\-]", "_", s.item.id)
        cit = s.citation
        cit_str = f"`{'; '.join(s.response.cited_inquiries)}`" if s.response.cited_inquiries else "*None*"
        sample_lines = [
            f"### Query `{s.item.id}` [{s.item.typology} / {s.item.complexity}]",
            f"**Question**: {s.item.question}",
            "",
            f"**Ground Truth Reference Inquiries**: `{'; '.join(s.item.inquiry_references)}`",
            f"**Retrieved Inquiries (Retriever)**: `{'; '.join(s.response.referenced_inquiries)}`",
            f"**Cited Inquiries (Generator)**: {cit_str}",
            "",
            f"- **Precision@K**: {s.retrieval.precision_at_k:.4f} | **Recall@K**: {s.retrieval.recall_at_k:.4f} | **MRR**: {s.retrieval.mrr:.4f}",
            (
                f"- **Citation Precision**: {cit.precision:.4f} | **Citation Recall**: {cit.recall:.4f} | **Hallucination Rate**: {cit.hallucination_rate:.4f}"
                if cit
                else "- **Citation Metrics**: *N/A*"
            ),
            f"- **ROUGE-L**: {s.lexical.rouge_l:.4f} | **Token F1**: {s.lexical.token_f1:.4f}",
            f"- **Faithfulness**: {s.llm_judge.faithfulness:.4f} | **Relevance**: {s.llm_judge.answer_relevance:.4f} | **Context Recall**: {s.llm_judge.context_recall:.4f}",
            "",
            f"**Generated Answer**:\n> {s.response.answer}",
            "",
            f"**Reference Answer**:\n> {s.item.reference_answer}",
            "",
        ]
        if question_rel_dir:
            sample_lines.extend(
                [
                    f"**Detailed Breakdown**: [View Detailed Question Report]({question_rel_dir}/{safe_id}.md)",
                    "",
                ]
            )
        sample_lines.extend(["---", ""])
        lines.extend(sample_lines)

    content = "\n".join(lines)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")

    return content


def generate_comparison_report(
    summary_a: EvaluationSummary,
    summary_b: EvaluationSummary,
    output_path: Path | str | None = None,
) -> str:
    """Generate a comparative benchmark report comparing two RAG pipelines."""
    ov_a = summary_a.overall
    ov_b = summary_b.overall

    def diff(val_b: float, val_a: float) -> str:
        d = val_b - val_a
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.4f}"

    lines = [
        f"# RAG Benchmark Architecture Comparison: `{summary_a.pipeline_name}` vs. `{summary_b.pipeline_name}`",
        "",
        "## 1. Overall Head-to-Head Comparison",
        "",
        f"| Metric | {summary_a.pipeline_name} (A) | {summary_b.pipeline_name} (B) | Absolute Delta (B - A) |",
        "|---|---|---|---|",
        f"| **Retrieval Precision@K** | {ov_a.mean_precision_at_k:.4f} | {ov_b.mean_precision_at_k:.4f} | {diff(ov_b.mean_precision_at_k, ov_a.mean_precision_at_k)} |",
        f"| **Retrieval Recall@K** | {ov_a.mean_recall_at_k:.4f} | {ov_b.mean_recall_at_k:.4f} | {diff(ov_b.mean_recall_at_k, ov_a.mean_recall_at_k)} |",
        f"| **Retrieval Hit Rate** | {ov_a.mean_hit_rate:.4f} | {ov_b.mean_hit_rate:.4f} | {diff(ov_b.mean_hit_rate, ov_a.mean_hit_rate)} |",
        f"| **Retrieval MRR** | {ov_a.mean_mrr:.4f} | {ov_b.mean_mrr:.4f} | {diff(ov_b.mean_mrr, ov_a.mean_mrr)} |",
        f"| **ROUGE-L** | {ov_a.mean_rouge_l:.4f} | {ov_b.mean_rouge_l:.4f} | {diff(ov_b.mean_rouge_l, ov_a.mean_rouge_l)} |",
        f"| **Token F1** | {ov_a.mean_token_f1:.4f} | {ov_b.mean_token_f1:.4f} | {diff(ov_b.mean_token_f1, ov_a.mean_token_f1)} |",
        f"| **Faithfulness (Judge)** | {ov_a.mean_faithfulness:.4f} | {ov_b.mean_faithfulness:.4f} | {diff(ov_b.mean_faithfulness, ov_a.mean_faithfulness)} |",
        f"| **Relevance (Judge)** | {ov_a.mean_answer_relevance:.4f} | {ov_b.mean_answer_relevance:.4f} | {diff(ov_b.mean_answer_relevance, ov_a.mean_answer_relevance)} |",
        f"| **Context Recall (Judge)** | {ov_a.mean_context_recall:.4f} | {ov_b.mean_context_recall:.4f} | {diff(ov_b.mean_context_recall, ov_a.mean_context_recall)} |",
        f"| **Mean Latency (ms)** | {ov_a.mean_latency_ms:.1f} | {ov_b.mean_latency_ms:.1f} | {diff(ov_b.mean_latency_ms, ov_a.mean_latency_ms)} |",
        "",
        "## 2. Typology Comparison (Recall@K / Faithfulness)",
        "",
        f"| Typology | {summary_a.pipeline_name} Recall | {summary_b.pipeline_name} Recall | {summary_a.pipeline_name} Faith | {summary_b.pipeline_name} Faith |",
        "|---|---|---|---|---|",
    ]

    all_typs = sorted(set(summary_a.by_typology.keys()).union(set(summary_b.by_typology.keys())))
    for typ in all_typs:
        ta = summary_a.by_typology.get(typ)
        tb = summary_b.by_typology.get(typ)
        rec_a = f"{ta.mean_recall_at_k:.4f}" if ta else "N/A"
        rec_b = f"{tb.mean_recall_at_k:.4f}" if tb else "N/A"
        f_a = f"{ta.mean_faithfulness:.4f}" if ta else "N/A"
        f_b = f"{tb.mean_faithfulness:.4f}" if tb else "N/A"
        lines.append(f"| **{typ}** | {rec_a} | {rec_b} | {f_a} | {f_b} |")

    lines.extend(
        [
            "",
            "## 3. Complexity Comparison (Recall@K / Faithfulness)",
            "",
            f"| Complexity | {summary_a.pipeline_name} Recall | {summary_b.pipeline_name} Recall | {summary_a.pipeline_name} Faith | {summary_b.pipeline_name} Faith |",
            "|---|---|---|---|---|",
        ]
    )

    all_comps = sorted(
        set(summary_a.by_complexity.keys()).union(set(summary_b.by_complexity.keys()))
    )
    for comp in all_comps:
        ca = summary_a.by_complexity.get(comp)
        cb = summary_b.by_complexity.get(comp)
        rec_a = f"{ca.mean_recall_at_k:.4f}" if ca else "N/A"
        rec_b = f"{cb.mean_recall_at_k:.4f}" if cb else "N/A"
        f_a = f"{ca.mean_faithfulness:.4f}" if ca else "N/A"
        f_b = f"{cb.mean_faithfulness:.4f}" if cb else "N/A"
        lines.append(f"| **{comp}** | {rec_a} | {rec_b} | {f_a} | {f_b} |")

    lines.append("")
    content = "\n".join(lines)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")

    return content


def generate_question_report(
    sample: SampleEvaluation, output_path: Path | str | None = None
) -> str:
    """Generate an individual, granular Markdown report for a single benchmark query.

    Includes exact prompts (System and User prompts for both Generator and LLM Judge),
    unabridged responses, comprehensive metric scorecard, and candidate chunks before
    and after reranking (with collapsible details for dropped candidates).
    """
    item = sample.item
    resp = sample.response
    ret = sample.retrieval
    lex = sample.lexical
    judge = sample.llm_judge
    cit = getattr(sample, "citation", None)

    typ_desc = TYPOLOGY_DESCRIPTIONS.get(item.typology, item.typology)
    comp_desc = COMPLEXITY_DESCRIPTIONS.get(item.complexity, item.complexity)
    reranker_name = resp.metadata.get("reranker") or "None (Direct Retrieval)"
    cited_str = f"`{'; '.join(resp.cited_inquiries)}`" if resp.cited_inquiries else "*None*"

    lines = [
        f"# Benchmark Sample Report: `{item.id}`",
        "",
        "## 1. Query Identification & Metadata",
        f"- **Sample ID**: `{item.id}`",
        f"- **Pipeline Architecture**: `{resp.pipeline_name}`",
        f"- **Investigative Typology**: `{item.typology}` — {typ_desc}",
        f"- **Complexity Level**: `{item.complexity}` — {comp_desc}",
        f"- **Ground Truth Target Inquiries**: `{'; '.join(item.inquiry_references)}`",
        f"- **Retrieved Document Inquiries (Retriever)**: `{'; '.join(resp.referenced_inquiries)}`",
        f"- **Cited Inquiries (Generator)**: {cited_str}",
        f"- **End-to-End Latency**: {resp.latency_ms:.1f} ms",
        f"- **Reranker Engine**: `{reranker_name}`",
        "",
        "## 2. Evaluation Metrics Scorecard",
        "",
        "| Metric Category | Metric | Score | Description |",
        "|---|---|---|---|",
        f"| Retrieval | Precision@K | {ret.precision_at_k:.4f} | Proportion of retrieved inquiries that are relevant |",
        f"| Retrieval | Recall@K | {ret.recall_at_k:.4f} | Coverage of target ground truth inquiry references |",
        f"| Retrieval | Hit Rate@K | {ret.hit_rate:.4f} | At least one relevant inquiry document identified |",
        f"| Retrieval | Mean Reciprocal Rank (MRR) | {ret.mrr:.4f} | Reciprocal rank of the first relevant inquiry document |",
        (
            f"| Citation Attribution | Citation Precision | {cit.precision:.4f} | Proportion of generator-cited inquiries present in ground truth |"
            if cit
            else "| Citation Attribution | Citation Precision | N/A | Generator citation precision |"
        ),
        (
            f"| Citation Attribution | Citation Recall | {cit.recall:.4f} | Coverage of target ground truth references cited by generator |"
            if cit
            else "| Citation Attribution | Citation Recall | N/A | Generator citation recall |"
        ),
        (
            f"| Citation Attribution | Citation Hallucination Rate | {cit.hallucination_rate:.4f} | Proportion of cited inquiries not grounded in retrieved context |"
            if cit
            else "| Citation Attribution | Citation Hallucination Rate | N/A | Generator citation hallucination rate |"
        ),
        f"| Lexical Overlap | ROUGE-L | {lex.rouge_l:.4f} | Longest common subsequence F1 against reference answer |",
        f"| Lexical Overlap | Token F1 | {lex.token_f1:.4f} | Harmonic mean of token precision and recall |",
        f"| Lexical Overlap | BLEU-4 | {lex.bleu_4:.4f} | 4-gram precision against ground truth reference |",
        f"| LLM-as-a-Judge | Faithfulness (Groundedness) | {judge.faithfulness:.4f} | Factual claims grounded strictly in retrieved context |",
        f"| LLM-as-a-Judge | Answer Relevance | {judge.answer_relevance:.4f} | Direct alignment with the investigative query intent |",
        f"| LLM-as-a-Judge | Context Recall | {judge.context_recall:.4f} | Presence of key investigative facts from reference answer |",
        "",
    ]

    if judge.critique:
        lines.extend(
            [
                "> **LLM Judge Critique & Justification**:",
                f"> {judge.critique}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. Investigative Query & Complete Answers",
            "",
            "### 3.1 Investigative Question",
            f"> {item.question}",
            "",
            "### 3.2 RAG Generated Answer (Full)",
            f"> {resp.answer}",
            "",
            "### 3.3 Ground Truth Reference Answer (Full)",
            f"> {item.reference_answer}",
            "",
            "## 4. Executed Prompts",
            "",
            "### 4.1 RAG Generator — System Prompt",
            "```text",
            resp.system_prompt or "*No system prompt recorded.*",
            "```",
            "",
            "### 4.2 RAG Generator — User Input Prompt (with Injected Context)",
            "```text",
            resp.prompt or "*No input prompt recorded.*",
            "```",
            "",
            "### 4.3 LLM Judge — System Prompt",
            "```text",
            judge.system_prompt or "*No judge system prompt recorded.*",
            "```",
            "",
            "### 4.4 LLM Judge — Evaluation Prompt",
            "```text",
            judge.prompt or "*No judge evaluation prompt recorded.*",
            "```",
            "",
            "## 5. Retrieval & Chunks Breakdown",
            "",
        ]
    )

    has_reranker = bool(resp.pre_rerank_chunks) and (resp.metadata.get("reranker") is not None)

    if has_reranker:
        pre_chunks = resp.pre_rerank_chunks
        final_chunks = resp.retrieved_chunks

        # Map final chunk IDs to their final rank (1-indexed) and rerank score
        final_chunk_info: dict[str, tuple[int, float]] = {
            res.chunk.id: (rank, res.score) for rank, res in enumerate(final_chunks, start=1)
        }

        lines.extend(
            [
                "### 5.1 Reranking Transition & Candidate Pool Overview",
                "",
                "| Final Rank | Initial Rank | Shift (Δ) | Document ID | Chunk ID | Page | Initial Score | Rerank Score | Status |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )

        for init_rank, res in enumerate(pre_chunks, start=1):
            cid = res.chunk.id
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            init_score = res.score

            if cid in final_chunk_info:
                f_rank, rerank_score = final_chunk_info[cid]
                delta = init_rank - f_rank
                delta_str = f"+{delta}" if delta > 0 else (f"{delta}" if delta < 0 else "=")
                status = "**Promoted to Context**"
                rerank_score_str = f"{rerank_score:.4f}"
                final_rank_str = f"#{f_rank}"
            else:
                final_rank_str = "*Dropped*"
                delta_str = "-"
                status = "*Dropped from Context*"
                rerank_score_str = "-"

            lines.append(
                f"| {final_rank_str} | #{init_rank} | {delta_str} | `{doc_id}` | `{cid}` | {page} | {init_score:.4f} | {rerank_score_str} | {status} |"
            )

        lines.extend(
            [
                "",
                "### 5.2 Final Chunks Promoted to Context",
                "",
            ]
        )

        for rank, res in enumerate(final_chunks, start=1):
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            init_rank = res.chunk.metadata.get("initial_retrieval_rank", "N/A")
            lines.extend(
                [
                    f"#### Final Chunk #{rank} — Document: `{doc_id}` (Page {page})",
                    f"- **Chunk ID**: `{res.chunk.id}`",
                    f"- **Rerank Score**: `{res.score:.4f}` | **Initial Rank**: `#{init_rank}`",
                    "",
                    "```text",
                    res.chunk.text,
                    "```",
                    "",
                ]
            )

        # 5.3 Dropped candidate chunks
        dropped_candidates = [
            (init_rank, res)
            for init_rank, res in enumerate(pre_chunks, start=1)
            if res.chunk.id not in final_chunk_info
        ]

        if dropped_candidates:
            lines.extend(
                [
                    "### 5.3 Dropped Candidate Chunks (Pre-Rerank Pool)",
                    "",
                    f"*A total of {len(dropped_candidates)} candidate chunks were evaluated during initial retrieval but excluded from the final context after reranking.*",
                    "",
                ]
            )
            for init_rank, res in dropped_candidates:
                doc_id = res.chunk.document_id
                page = res.chunk.metadata.get("page_number", "N/A")
                lines.extend(
                    [
                        "<details>",
                        f"<summary><strong>Candidate #{init_rank} (Dropped)</strong> — Document: <code>{doc_id}</code> (Page {page}) | Initial Score: {res.score:.4f} | Chunk ID: <code>{res.chunk.id}</code></summary>",
                        "",
                        "```text",
                        res.chunk.text,
                        "```",
                        "",
                        "</details>",
                        "",
                    ]
                )
    else:
        # No reranker configured
        lines.extend(
            [
                "> [!NOTE]",
                "> **Reranker was not configured for this pipeline.** Chunks below were retrieved directly via Top-K retrieval and forwarded to the generator context.",
                "",
                "### 5.1 Retrieved Context Chunks",
                "",
                "| Rank | Document ID | Chunk ID | Page | Retrieval Score |",
                "|---|---|---|---|---|",
            ]
        )
        for rank, res in enumerate(resp.retrieved_chunks, start=1):
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            lines.append(f"| #{rank} | `{doc_id}` | `{res.chunk.id}` | {page} | {res.score:.4f} |")

        lines.extend(
            [
                "",
                "### 5.2 Context Evidence Chunks (Fed to Generator)",
                "",
            ]
        )
        for rank, res in enumerate(resp.retrieved_chunks, start=1):
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            lines.extend(
                [
                    f"#### Chunk #{rank} — Document: `{doc_id}` (Page {page})",
                    f"- **Chunk ID**: `{res.chunk.id}`",
                    f"- **Retrieval Score**: `{res.score:.4f}`",
                    "",
                    "```text",
                    res.chunk.text,
                    "```",
                    "",
                ]
            )

    content = "\n".join(lines)
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")

    return content


def generate_all_question_reports(summary: EvaluationSummary, output_dir: Path | str) -> list[Path]:
    """Generate detailed individual Markdown reports for each benchmark question.

    Args:
        summary: The complete EvaluationSummary.
        output_dir: Destination directory path (e.g. output_dir / "questions").

    Returns:
        List of generated file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for s in summary.samples:
        safe_id = re.sub(r"[^\w\-]", "_", s.item.id)
        report_file = out_dir / f"{safe_id}.md"
        generate_question_report(s, output_path=report_file)
        generated.append(report_file)

    return generated
