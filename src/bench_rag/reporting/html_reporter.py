"""HTML reporting module for forensic RAG benchmark evaluations.

Generates self-contained, responsive, professionally styled static HTML reports
with zero external dependencies (air-gapped and offline ready).
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from bench_rag.core.evaluator import EvaluationSummary, SampleEvaluation, TaxonomyAggregate

TYPOLOGY_DESCRIPTIONS = {
    "T1": "T1 (Identification &amp; Associations): Intermediaries, shell entities, strawmen (<em>laranjas</em>), corporate ownership trees, beneficial owners.",
    "T2": "T2 (Criminal Dynamics): Modus operandi, money laundering typologies, criminal jargon, asset concealment tactics.",
    "T3": "T3 (Tracking &amp; Chronology): Asset flow trails, wire transfer sequences, chronological timeline of suspicious activities.",
}

COMPLEXITY_DESCRIPTIONS = {
    "C1": "C1 (Low / Factual): Single-document, literal fact extraction.",
    "C2": "C2 (Medium / Relational): Multi-document aggregation testing retrieval exhaustiveness.",
    "C3": "C3 (High / Analytical): Broad inferential context regarding criminal intent (<em>mens rea</em> / <em>dolo</em>).",
}

CSS_STYLES = """
:root {
  --bg: #f8fafc;
  --card-bg: #ffffff;
  --text-main: #0f172a;
  --text-muted: #475569;
  --border: #e2e8f0;
  --border-strong: #cbd5e1;
  --primary: #1e40af;
  --primary-light: #eff6ff;
  --primary-border: #bfdbfe;
  --success: #166534;
  --success-bg: #dcfce7;
  --danger: #991b1b;
  --danger-bg: #fee2e2;
  --warning: #92400e;
  --warning-bg: #fef3c7;
  --code-bg: #f1f5f9;
  --table-head: #f8fafc;
  --table-alt: #fdfdfe;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
}

*, *::before, *::after {
  box-sizing: border-box;
}

body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--text-main);
  background-color: var(--bg);
  line-height: 1.6;
  font-size: 15px;
  -webkit-font-smoothing: antialiased;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 4rem 1.5rem;
}

header {
  border-bottom: 2px solid var(--border);
  padding-bottom: 1.5rem;
  margin-bottom: 2rem;
}

h1, h2, h3, h4 {
  color: var(--text-main);
  font-weight: 700;
  line-height: 1.3;
}

h1 {
  font-size: 2rem;
  margin: 0 0 0.5rem 0;
}

h2 {
  font-size: 1.4rem;
  margin: 2.5rem 0 1rem 0;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}

h3 {
  font-size: 1.15rem;
  margin: 1.5rem 0 0.75rem 0;
}

h4 {
  font-size: 1rem;
  margin: 1.25rem 0 0.5rem 0;
}

p {
  margin: 0 0 1rem 0;
}

a {
  color: var(--primary);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.back-link {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--primary);
  background: var(--primary-light);
  border: 1px solid var(--primary-border);
  padding: 0.4rem 0.85rem;
  border-radius: 6px;
  margin-bottom: 1.5rem;
  transition: background 0.15s ease-in-out;
}

.back-link:hover {
  background: #dbeafe;
  text-decoration: none;
}

/* Cards & Grid Layout */
.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.25rem;
  box-shadow: var(--shadow-sm);
}

.card-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 600;
  color: var(--text-muted);
  margin-bottom: 0.35rem;
}

.card-value {
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text-main);
  margin: 0;
}

.card-subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

/* Callouts */
.callout {
  padding: 1rem 1.25rem;
  border-radius: 6px;
  border-left: 4px solid var(--primary);
  background: var(--primary-light);
  margin-bottom: 1.5rem;
}

.callout.warning {
  border-left-color: var(--warning);
  background: var(--warning-bg);
}

.callout.success {
  border-left-color: var(--success);
  background: var(--success-bg);
}

.callout p:last-child {
  margin-bottom: 0;
}

/* Badges */
.badge {
  display: inline-block;
  padding: 0.2rem 0.55rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.025em;
  text-transform: uppercase;
}

.badge-primary {
  background: var(--primary-light);
  color: var(--primary);
  border: 1px solid var(--primary-border);
}

.badge-success {
  background: var(--success-bg);
  color: var(--success);
}

.badge-danger {
  background: var(--danger-bg);
  color: var(--danger);
}

.badge-neutral {
  background: var(--code-bg);
  color: var(--text-muted);
  border: 1px solid var(--border);
}

/* Tables */
.table-wrapper {
  overflow-x: auto;
  margin-bottom: 2rem;
  border: 1px solid var(--border);
  border-radius: 8px;
  background: var(--card-bg);
  box-shadow: var(--shadow-sm);
}

table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
  font-size: 0.9rem;
}

thead {
  background: var(--table-head);
  border-bottom: 2px solid var(--border);
}

th {
  padding: 0.75rem 1rem;
  font-weight: 600;
  color: var(--text-main);
  white-space: nowrap;
}

td {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
  color: var(--text-main);
}

tbody tr:last-child td {
  border-bottom: none;
}

tbody tr:nth-child(even) {
  background: var(--table-alt);
}

.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

/* Pre & Code */
pre, code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

code {
  font-size: 0.875em;
  padding: 0.15rem 0.35rem;
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 4px;
}

pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 0.85rem;
  line-height: 1.5;
  margin: 0.5rem 0 1.25rem 0;
}

pre code {
  background: transparent;
  padding: 0;
  border: none;
}

/* Blockquotes / Answers */
.answer-box {
  background: var(--card-bg);
  border-left: 4px solid var(--primary);
  border-radius: 0 6px 6px 0;
  padding: 1rem 1.25rem;
  margin: 0.5rem 0 1.25rem 0;
  border-top: 1px solid var(--border);
  border-right: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}

.answer-box.reference {
  border-left-color: var(--success);
}

/* Details / Collapsible */
details {
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card-bg);
  margin-bottom: 0.75rem;
  overflow: hidden;
}

summary {
  padding: 0.75rem 1rem;
  background: var(--table-head);
  cursor: pointer;
  font-weight: 600;
  font-size: 0.9rem;
  user-select: none;
  outline: none;
  list-style: none;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

summary::-webkit-details-marker {
  display: none;
}

summary::after {
  content: "▾";
  font-size: 0.85rem;
  color: var(--text-muted);
  transition: transform 0.2s ease;
}

details[open] summary::after {
  transform: rotate(180deg);
}

details .details-content {
  padding: 1rem;
  border-top: 1px solid var(--border);
}

/* Metric Guide */
details.metric-guide {
  margin: 1.25rem 0 2rem 0;
  border: 1px solid var(--primary-border);
  background: var(--card-bg);
  border-radius: 8px;
  box-shadow: var(--shadow-sm);
}

details.metric-guide summary {
  background: var(--primary-light);
  color: var(--primary);
  font-size: 0.95rem;
  padding: 0.85rem 1.25rem;
  border-bottom: none;
}

details.metric-guide summary:hover {
  background: #e0edff;
}

details.metric-guide[open] summary {
  border-bottom: 1px solid var(--border);
}

details.metric-guide .details-content {
  padding: 1.25rem 1.25rem 0.5rem 1.25rem;
  background: var(--card-bg);
}

.metric-guide-group {
  margin-bottom: 1.5rem;
}

.metric-guide-group h4 {
  margin: 0 0 0.75rem 0;
  font-size: 0.95rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.metric-guide-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
  margin-bottom: 0.5rem;
}

.metric-guide-table tr {
  border-bottom: 1px solid var(--border);
}

.metric-guide-table tr:last-child {
  border-bottom: none;
}

.metric-guide-table td {
  padding: 0.5rem 0.75rem;
  vertical-align: top;
}

.metric-guide-table td.metric-name {
  font-weight: 600;
  width: 260px;
  white-space: nowrap;
  color: var(--text-main);
}

.metric-guide-table td.metric-desc {
  color: var(--text-muted);
  line-height: 1.5;
}

/* Sample Card in Overview */
.sample-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.5rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
}

.sample-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.75rem;
  margin-bottom: 1rem;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.meta-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.5rem 1.5rem;
  margin-bottom: 1rem;
}

.meta-item {
  font-size: 0.9rem;
}

.meta-label {
  font-weight: 600;
  color: var(--text-muted);
}

/* Print Styles */
@media print {
  body {
    background: #ffffff;
    font-size: 11pt;
  }
  .container {
    max-width: 100%;
    padding: 0;
  }
  .back-link, details summary::after {
    display: none !important;
  }
  .card, .table-wrapper, .sample-card, pre, details {
    box-shadow: none !important;
    border-color: #cbd5e1 !important;
    page-break-inside: avoid;
  }
  details[open] {
    display: block;
  }
}
"""


def _html_page(title: str, body: str) -> str:
    """Wrap content in a standalone, valid HTML5 template with embedded CSS."""
    clean_title = html.escape(title)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{clean_title}</title>
  <style>
{CSS_STYLES}
  </style>
</head>
<body>
  <div class="container">
{body}
  </div>
</body>
</html>
"""


def _format_aggregate_table(title: str, aggregates: dict[str, TaxonomyAggregate]) -> str:
    """Format a styled HTML table from category -> TaxonomyAggregate mapping."""
    rows: list[str] = []
    for cat, agg in aggregates.items():
        cat_esc = html.escape(str(cat))
        ans_corr = getattr(agg, "mean_answer_correctness", 0.0)
        rows.append(
            f"<tr>"
            f"<td><strong>{cat_esc}</strong></td>"
            f"<td class='num'>{agg.sample_count}</td>"
            f"<td class='num'>{agg.mean_precision_at_k:.4f}</td>"
            f"<td class='num'>{agg.mean_recall_at_k:.4f}</td>"
            f"<td class='num'>{agg.mean_hit_rate:.4f}</td>"
            f"<td class='num'>{agg.mean_mrr:.4f}</td>"
            f"<td class='num'>{agg.mean_citation_precision:.4f}</td>"
            f"<td class='num'>{agg.mean_citation_recall:.4f}</td>"
            f"<td class='num'>{agg.mean_rouge_l:.4f}</td>"
            f"<td class='num'>{agg.mean_token_f1:.4f}</td>"
            f"<td class='num'>{agg.mean_faithfulness:.4f}</td>"
            f"<td class='num'>{agg.mean_answer_relevance:.4f}</td>"
            f"<td class='num'>{agg.mean_context_recall:.4f}</td>"
            f"<td class='num'>{ans_corr:.4f}</td>"
            f"<td class='num'>{agg.mean_latency_ms:.1f}</td>"
            f"</tr>"
        )

    return f"""
<h3>{html.escape(title)}</h3>
<div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>Category</th>
        <th class="num">Samples</th>
        <th class="num">Precision@K</th>
        <th class="num">Recall@K</th>
        <th class="num">Hit Rate</th>
        <th class="num">MRR</th>
        <th class="num">Cit. Prec.</th>
        <th class="num">Cit. Rec.</th>
        <th class="num">ROUGE-L</th>
        <th class="num">Token F1</th>
        <th class="num">Faithfulness</th>
        <th class="num">Relevance</th>
        <th class="num">Context Recall</th>
        <th class="num">Answer Correctness</th>
        <th class="num">Latency (ms)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</div>
"""



def _render_metric_guide_html() -> str:
    """Render a collapsible, concise executive guide for evaluation metrics (Sections 1.1 - 1.4)."""
    return """<details class="metric-guide">
  <summary>
    <span>📘 <strong>Evaluation Metrics Interpretation Guide</strong> &mdash; Concise definitions and investigative relevance</span>
  </summary>
  <div class="details-content">
    <div class="metric-guide-group">
      <h4>Retrieval Metrics</h4>
      <table class="metric-guide-table">
        <tbody>
          <tr>
            <td class="metric-name">Precision@K</td>
            <td class="metric-desc">Proportion of retrieved chunks in the top-<em>k</em> that belong to relevant ground-truth inquiries, assessing context signal-to-noise ratio fed to the generator.</td>
          </tr>
          <tr>
            <td class="metric-name">Recall@K</td>
            <td class="metric-desc">Proportion of expected ground-truth documents successfully retrieved in the top-<em>k</em>; critical for multi-document investigations where missing records compromises factual completeness.</td>
          </tr>
          <tr>
            <td class="metric-name">Hit Rate@K</td>
            <td class="metric-desc">Binary indicator (1 if at least one relevant document is retrieved, 0 otherwise) confirming baseline evidentiary retrieval success.</td>
          </tr>
          <tr>
            <td class="metric-name">Mean Reciprocal Rank (MRR)</td>
            <td class="metric-desc">Reciprocal rank of the first relevant chunk (1 / rank), rewarding pipelines that position key evidence at the very top of the ranking.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="metric-guide-group">
      <h4>Citation Attribution Metrics</h4>
      <table class="metric-guide-table">
        <tbody>
          <tr>
            <td class="metric-name">Citation Precision</td>
            <td class="metric-desc">Proportion of inquiry documents and sources explicitly cited in the model's answer that are present in the ground-truth references.</td>
          </tr>
          <tr>
            <td class="metric-name">Citation Recall</td>
            <td class="metric-desc">Coverage of target ground-truth reference documents explicitly cited in the generated answer's reasoning.</td>
          </tr>
          <tr>
            <td class="metric-name">Citation Hallucination Rate</td>
            <td class="metric-desc">Percentage of cited documents that <strong>were not present in the retrieved context</strong>, identifying critical forensic source hallucinations.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="metric-guide-group">
      <h4>Lexical Overlap Metrics</h4>
      <table class="metric-guide-table">
        <tbody>
          <tr>
            <td class="metric-name">Token F1</td>
            <td class="metric-desc">Harmonic mean of token-level precision and recall between the generated response and the reference answer, measuring preservation of essential keywords and vocabulary.</td>
          </tr>
          <tr>
            <td class="metric-name">ROUGE-L</td>
            <td class="metric-desc">Symmetric harmonic F1 score (&beta; = 1.0) based on the Longest Common Subsequence (LCS), capturing sentence-level structural coherence and word ordering against the reference answer.</td>
          </tr>
          <tr>
            <td class="metric-name">BLEU-4</td>
            <td class="metric-desc">Geometric mean of 1-to-4-gram precision with brevity penalty and smoothing, reflecting phrase-level fluency and exact textual alignment.</td>
          </tr>
        </tbody>
      </table>
    </div>

    <div class="metric-guide-group" style="margin-bottom: 0.5rem;">
      <h4>LLM-as-a-Judge Forensic Metrics (0.0 &ndash; 1.0)</h4>
      <table class="metric-guide-table">
        <tbody>
          <tr>
            <td class="metric-name">Faithfulness (Groundedness)</td>
            <td class="metric-desc">Evaluates whether every factual claim in the generated answer is strictly grounded in and substantiated by the retrieved context, serving as the primary safeguard against hallucinations.</td>
          </tr>
          <tr>
            <td class="metric-name">Answer Relevance</td>
            <td class="metric-desc">Assesses whether the generated response directly and comprehensively answers the investigative query, penalizing evasive, repetitive, or off-topic content.</td>
          </tr>
          <tr>
            <td class="metric-name">Context Recall</td>
            <td class="metric-desc">Verifies whether the retrieved context chunks contain all essential investigative facts present in the official reference answer (RAGAS Context Recall).</td>
          </tr>
          <tr>
            <td class="metric-name">Answer Correctness</td>
            <td class="metric-desc">Verifies whether the generated answer captures all essential investigative facts present in the official reference answer.</td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</details>"""


def _render_pipeline_config_html(summary: EvaluationSummary) -> str:
    """Render the essential pipeline architecture and models configuration section."""
    meta = getattr(summary, "execution_metadata", None) or {}

    p_arch = meta.get("pipeline_type")
    if not p_arch:
        name_lower = summary.pipeline_name.lower()
        p_arch = (
            "Hybrid BM25 + Dense RAG"
            if "hybrid" in name_lower
            else "Standard Dense RAG"
        )

    top_k = meta.get("top_k")
    if top_k is None:
        if (
            summary.samples
            and summary.samples[0].response.metadata.get("top_k") is not None
        ):
            top_k = summary.samples[0].response.metadata.get("top_k")
        else:
            top_k = 5

    gen_model = meta.get("generator_model")
    if not gen_model or gen_model == "N/A":
        if summary.samples:
            gen_model = summary.samples[0].response.metadata.get("model_name") or "N/A"
        else:
            gen_model = "N/A"

    emb_model = meta.get("embedding_model") or "N/A"

    rerank_model = meta.get("reranker_model")
    if not rerank_model:
        if summary.samples:
            r = summary.samples[0].response.metadata.get("reranker")
            rerank_model = r or "None (Direct Retrieval)"
        else:
            rerank_model = "None (Direct Retrieval)"

    judge_model = meta.get("judge_model") or "N/A"

    return f"""<div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>Component</th>
        <th>Configuration / Model</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Pipeline Architecture</strong></td><td><code>{html.escape(str(p_arch))}</code></td></tr>
      <tr><td><strong>Top-K Retrieval</strong></td><td><code>{html.escape(str(top_k))}</code></td></tr>
      <tr><td><strong>Generator Model</strong></td><td><code>{html.escape(str(gen_model))}</code></td></tr>
      <tr><td><strong>Embedding Model</strong></td><td><code>{html.escape(str(emb_model))}</code></td></tr>
      <tr><td><strong>Reranker Model</strong></td><td><code>{html.escape(str(rerank_model))}</code></td></tr>
      <tr><td><strong>LLM Judge Model</strong></td><td><code>{html.escape(str(judge_model))}</code></td></tr>
    </tbody>
  </table>
</div>"""


def _render_comparison_config_html(
    summary_a: EvaluationSummary, summary_b: EvaluationSummary
) -> str:
    """Render side-by-side comparison of pipeline architecture and models configuration."""

    def extract_cfg(s: EvaluationSummary) -> dict[str, Any]:
        meta = getattr(s, "execution_metadata", None) or {}
        p_arch = meta.get("pipeline_type")
        if not p_arch:
            p_arch = (
                "Hybrid BM25 + Dense RAG"
                if "hybrid" in s.pipeline_name.lower()
                else "Standard Dense RAG"
            )

        top_k = meta.get("top_k")
        if top_k is None:
            if (
                s.samples
                and s.samples[0].response.metadata.get("top_k") is not None
            ):
                top_k = s.samples[0].response.metadata.get("top_k")
            else:
                top_k = 5

        gen = meta.get("generator_model")
        if not gen or gen == "N/A":
            gen = (
                s.samples[0].response.metadata.get("model_name", "N/A")
                if s.samples
                else "N/A"
            )

        emb = meta.get("embedding_model") or "N/A"

        rerank = meta.get("reranker_model")
        if not rerank:
            if s.samples:
                r = s.samples[0].response.metadata.get("reranker")
                rerank = r or "None (Direct Retrieval)"
            else:
                rerank = "None (Direct Retrieval)"

        judge = meta.get("judge_model") or "N/A"

        return {
            "pipeline_type": p_arch,
            "top_k": top_k,
            "generator_model": gen,
            "embedding_model": emb,
            "reranker_model": rerank,
            "judge_model": judge,
        }

    cfg_a = extract_cfg(summary_a)
    cfg_b = extract_cfg(summary_b)
    name_a = html.escape(summary_a.pipeline_name)
    name_b = html.escape(summary_b.pipeline_name)

    return f"""<div class="table-wrapper">
  <table>
    <thead>
      <tr>
        <th>Component</th>
        <th class="num">{name_a} (A)</th>
        <th class="num">{name_b} (B)</th>
      </tr>
    </thead>
    <tbody>
      <tr><td><strong>Pipeline Architecture</strong></td><td class="num"><code>{html.escape(str(cfg_a['pipeline_type']))}</code></td><td class="num"><code>{html.escape(str(cfg_b['pipeline_type']))}</code></td></tr>
      <tr><td><strong>Top-K Retrieval</strong></td><td class="num"><code>{html.escape(str(cfg_a['top_k']))}</code></td><td class="num"><code>{html.escape(str(cfg_b['top_k']))}</code></td></tr>
      <tr><td><strong>Generator Model</strong></td><td class="num"><code>{html.escape(str(cfg_a['generator_model']))}</code></td><td class="num"><code>{html.escape(str(cfg_b['generator_model']))}</code></td></tr>
      <tr><td><strong>Embedding Model</strong></td><td class="num"><code>{html.escape(str(cfg_a['embedding_model']))}</code></td><td class="num"><code>{html.escape(str(cfg_b['embedding_model']))}</code></td></tr>
      <tr><td><strong>Reranker Model</strong></td><td class="num"><code>{html.escape(str(cfg_a['reranker_model']))}</code></td><td class="num"><code>{html.escape(str(cfg_b['reranker_model']))}</code></td></tr>
      <tr><td><strong>LLM Judge Model</strong></td><td class="num"><code>{html.escape(str(cfg_a['judge_model']))}</code></td><td class="num"><code>{html.escape(str(cfg_b['judge_model']))}</code></td></tr>
    </tbody>
  </table>
</div>"""


def generate_html_report(
    summary: EvaluationSummary,
    output_path: Path | str | None = None,
    question_rel_dir: str | None = "questions",
) -> str:
    """Generate a comprehensive, self-contained HTML evaluation report."""
    ov = summary.overall
    p_name = html.escape(summary.pipeline_name)
    ov_ans_corr = getattr(ov, "mean_answer_correctness", 0.0)

    body_parts = [
        "<header>",
        f"  <h1>RAG Benchmark Evaluation Report: <code>{p_name}</code></h1>",
        "  <p class='card-subtitle'>Complete benchmark analysis across investigative taxonomies, retrieval depth, citation attribution, and LLM-as-a-judge scoring.</p>",
        "</header>",
        "<h2>1. Executive Summary</h2>",
        "<div class='card-grid'>",
        f"  <div class='card'><div class='card-title'>Architecture</div><div class='card-value' style='font-size: 1.25rem;'>{p_name}</div><div class='card-subtitle'>Evaluated Pipeline</div></div>",
        f"  <div class='card'><div class='card-title'>Samples</div><div class='card-value'>{summary.total_samples}</div><div class='card-subtitle'>Ground Truth Queries</div></div>",
        f"  <div class='card'><div class='card-title'>Retrieval Recall@K</div><div class='card-value'>{ov.mean_recall_at_k:.4f}</div><div class='card-subtitle'>Target Inquiries Found</div></div>",
        f"  <div class='card'><div class='card-title'>Mean MRR</div><div class='card-value'>{ov.mean_mrr:.4f}</div><div class='card-subtitle'>Reciprocal Rank</div></div>",
        f"  <div class='card'><div class='card-title'>Faithfulness</div><div class='card-value'>{ov.mean_faithfulness:.4f}</div><div class='card-subtitle'>LLM Judge Groundedness</div></div>",
        f"  <div class='card'><div class='card-title'>Answer Relevance</div><div class='card-value'>{ov.mean_answer_relevance:.4f}</div><div class='card-subtitle'>Query Intent Alignment</div></div>",
        f"  <div class='card'><div class='card-title'>Context Recall</div><div class='card-value'>{ov.mean_context_recall:.4f}</div><div class='card-subtitle'>Context Coverage</div></div>",
        f"  <div class='card'><div class='card-title'>Answer Correctness</div><div class='card-value'>{ov_ans_corr:.4f}</div><div class='card-subtitle'>Reference Answer Coverage</div></div>",
        f"  <div class='card'><div class='card-title'>Citation Precision</div><div class='card-value'>{ov.mean_citation_precision:.4f}</div><div class='card-subtitle'>Source Accuracy</div></div>",
        f"  <div class='card'><div class='card-title'>Mean Latency</div><div class='card-value'>{ov.mean_latency_ms:.1f} <span style='font-size:0.9rem; font-weight:normal;'>ms</span></div><div class='card-subtitle'>End-to-End Pipeline</div></div>",
        "</div>",
        "<h2>2. Pipeline &amp; Models Configuration</h2>",
        _render_pipeline_config_html(summary),
        "<h2>3. Global Performance Metrics</h2>",
        "<div class='table-wrapper'>",
        "  <table>",
        "    <thead><tr><th>Metric Category</th><th>Metric</th><th class='num'>Score</th></tr></thead>",
        "    <tbody>",
        f"      <tr><td>Retrieval</td><td>Precision@K</td><td class='num'>{ov.mean_precision_at_k:.4f}</td></tr>",
        f"      <tr><td>Retrieval</td><td>Recall@K</td><td class='num'>{ov.mean_recall_at_k:.4f}</td></tr>",
        f"      <tr><td>Retrieval</td><td>Hit Rate@K</td><td class='num'>{ov.mean_hit_rate:.4f}</td></tr>",
        f"      <tr><td>Retrieval</td><td>Mean Reciprocal Rank (MRR)</td><td class='num'>{ov.mean_mrr:.4f}</td></tr>",
        f"      <tr><td>Citation Attribution</td><td>Citation Precision</td><td class='num'>{ov.mean_citation_precision:.4f}</td></tr>",
        f"      <tr><td>Citation Attribution</td><td>Citation Recall</td><td class='num'>{ov.mean_citation_recall:.4f}</td></tr>",
        f"      <tr><td>Citation Attribution</td><td>Citation Hallucination Rate</td><td class='num'>{ov.mean_citation_hallucination_rate:.4f}</td></tr>",
        f"      <tr><td>Lexical Overlap</td><td>ROUGE-L</td><td class='num'>{ov.mean_rouge_l:.4f}</td></tr>",
        f"      <tr><td>Lexical Overlap</td><td>Token F1</td><td class='num'>{ov.mean_token_f1:.4f}</td></tr>",
        f"      <tr><td>Lexical Overlap</td><td>BLEU-4</td><td class='num'>{ov.mean_bleu_4:.4f}</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Faithfulness (Groundedness)</td><td class='num'>{ov.mean_faithfulness:.4f}</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Answer Relevance</td><td class='num'>{ov.mean_answer_relevance:.4f}</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Context Recall</td><td class='num'>{ov.mean_context_recall:.4f}</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Answer Correctness</td><td class='num'>{ov_ans_corr:.4f}</td></tr>",
        f"      <tr><td>Performance</td><td>Mean Latency</td><td class='num'>{ov.mean_latency_ms:.1f} ms</td></tr>",
        "    </tbody>",
        "  </table>",
        "</div>",
        _render_metric_guide_html(),
        "<h2>4. Performance by Investigative Typology</h2>",
        "<ul>",
        f"  <li>{TYPOLOGY_DESCRIPTIONS['T1']}</li>",
        f"  <li>{TYPOLOGY_DESCRIPTIONS['T2']}</li>",
        f"  <li>{TYPOLOGY_DESCRIPTIONS['T3']}</li>",
        "</ul>",
        _format_aggregate_table("Typology Breakdown", summary.by_typology),
        "<h2>5. Performance by Complexity Level</h2>",
        "<ul>",
        f"  <li>{COMPLEXITY_DESCRIPTIONS['C1']}</li>",
        f"  <li>{COMPLEXITY_DESCRIPTIONS['C2']}</li>",
        f"  <li>{COMPLEXITY_DESCRIPTIONS['C3']}</li>",
        "</ul>",
        _format_aggregate_table("Complexity Breakdown", summary.by_complexity),
        "<h2>6. Typology x Complexity Matrix</h2>",
        "<div class='table-wrapper'>",
        "  <table>",
        "    <thead><tr><th>Typology</th><th>Complexity</th><th class='num'>Samples</th><th class='num'>Recall@K</th><th class='num'>MRR</th><th class='num'>Token F1</th><th class='num'>Faithfulness</th><th class='num'>Relevance</th><th class='num'>Context Recall</th><th class='num'>Answer Correctness</th></tr></thead>",
        "    <tbody>",
    ]

    for typ, comp_dict in sorted(summary.matrix.items()):
        for comp, agg in sorted(comp_dict.items()):
            ans_corr_m = getattr(agg, "mean_answer_correctness", 0.0)
            body_parts.append(
                f"<tr>"
                f"<td><strong>{html.escape(typ)}</strong></td>"
                f"<td><strong>{html.escape(comp)}</strong></td>"
                f"<td class='num'>{agg.sample_count}</td>"
                f"<td class='num'>{agg.mean_recall_at_k:.4f}</td>"
                f"<td class='num'>{agg.mean_mrr:.4f}</td>"
                f"<td class='num'>{agg.mean_token_f1:.4f}</td>"
                f"<td class='num'>{agg.mean_faithfulness:.4f}</td>"
                f"<td class='num'>{agg.mean_answer_relevance:.4f}</td>"
                f"<td class='num'>{agg.mean_context_recall:.4f}</td>"
                f"<td class='num'>{ans_corr_m:.4f}</td>"
                f"</tr>"
            )

    body_parts.extend(
        [
            "    </tbody>",
            "  </table>",
            "</div>",
            "<h2>7. Sample-Level Granular Results</h2>",
        ]
    )

    for s in summary.samples:
        safe_id = re.sub(r"[^\w\-]", "_", s.item.id)
        cit = s.citation
        cit_str = html.escape("; ".join(s.response.cited_inquiries)) if s.response.cited_inquiries else "<em>None</em>"
        ref_str = html.escape("; ".join(s.item.inquiry_references)) if s.item.inquiry_references else "<em>None</em>"
        ret_str = html.escape("; ".join(s.response.referenced_inquiries)) if s.response.referenced_inquiries else "<em>None</em>"
        s_ans_corr = getattr(s.llm_judge, "answer_correctness", 0.0)

        q_link_html = ""
        if question_rel_dir:
            q_href = f"{question_rel_dir}/{safe_id}.html"
            q_link_html = f"<div style='margin-top: 1rem;'><a href='{q_href}' class='back-link' style='margin-bottom:0;'>View Detailed Question Report &rarr;</a></div>"

        cit_metrics_html = (
            f"Cit. Prec: <strong>{cit.precision:.4f}</strong> | Cit. Rec: <strong>{cit.recall:.4f}</strong> | Hallucination Rate: <strong>{cit.hallucination_rate:.4f}</strong>"
            if cit
            else "Cit. Metrics: <em>N/A</em>"
        )

        body_parts.extend(
            [
                "<div class='sample-card'>",
                "  <div class='sample-card-header'>",
                f"    <h3 style='margin:0;'>Query <code>{html.escape(s.item.id)}</code></h3>",
                f"    <div><span class='badge badge-primary'>{html.escape(s.item.typology)}</span> <span class='badge badge-neutral'>{html.escape(s.item.complexity)}</span></div>",
                "  </div>",
                f"  <p><strong>Question</strong>: {html.escape(s.item.question)}</p>",
                "  <div class='meta-list'>",
                f"    <div class='meta-item'><span class='meta-label'>Ground Truth Targets:</span> <code>{ref_str}</code></div>",
                f"    <div class='meta-item'><span class='meta-label'>Retrieved Inquiries:</span> <code>{ret_str}</code></div>",
                f"    <div class='meta-item'><span class='meta-label'>Cited by Generator:</span> <code>{cit_str}</code></div>",
                "  </div>",
                "  <div style='background: var(--code-bg); padding: 0.6rem 0.85rem; border-radius: 6px; font-size: 0.85rem; margin-bottom: 1rem; border: 1px solid var(--border);'>",
                f"    Precision@K: <strong>{s.retrieval.precision_at_k:.4f}</strong> | Recall@K: <strong>{s.retrieval.recall_at_k:.4f}</strong> | MRR: <strong>{s.retrieval.mrr:.4f}</strong> | {cit_metrics_html}<br>",
                f"    ROUGE-L: <strong>{s.lexical.rouge_l:.4f}</strong> | Token F1: <strong>{s.lexical.token_f1:.4f}</strong> | BLEU-4: <strong>{s.lexical.bleu_4:.4f}</strong> | Faithfulness: <strong>{s.llm_judge.faithfulness:.4f}</strong> | Relevance: <strong>{s.llm_judge.answer_relevance:.4f}</strong> | Context Recall: <strong>{s.llm_judge.context_recall:.4f}</strong> | Answer Correctness: <strong>{s_ans_corr:.4f}</strong>",
                "  </div>",
                "  <strong>Generated Answer:</strong>",
                f"  <div class='answer-box'>{html.escape(s.response.answer)}</div>",
                "  <strong>Reference Answer:</strong>",
                f"  <div class='answer-box reference'>{html.escape(s.item.reference_answer)}</div>",
                f"  {q_link_html}",
                "</div>",
            ]
        )

    content = _html_page(title=f"RAG Evaluation: {summary.pipeline_name}", body="\n".join(body_parts))
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")

    return content


def generate_question_report(
    sample: SampleEvaluation,
    output_path: Path | str | None = None,
    main_report_href: str | None = "../evaluation_report.html",
) -> str:
    """Generate an individual, granular HTML report for a single benchmark query."""
    item = sample.item
    resp = sample.response
    ret = sample.retrieval
    lex = sample.lexical
    judge = sample.llm_judge
    cit = getattr(sample, "citation", None)
    ans_corr = getattr(judge, "answer_correctness", 0.0)

    typ_desc = TYPOLOGY_DESCRIPTIONS.get(item.typology, item.typology)
    comp_desc = COMPLEXITY_DESCRIPTIONS.get(item.complexity, item.complexity)
    reranker_name = resp.metadata.get("reranker") or "None (Direct Retrieval)"
    cited_str = html.escape("; ".join(resp.cited_inquiries)) if resp.cited_inquiries else "<em>None</em>"
    ref_str = html.escape("; ".join(item.inquiry_references)) if item.inquiry_references else "<em>None</em>"
    ret_str = html.escape("; ".join(resp.referenced_inquiries)) if resp.referenced_inquiries else "<em>None</em>"

    back_button = (
        f"<a href='{main_report_href}' class='back-link'>&larr; Back to Evaluation Report</a>"
        if main_report_href
        else ""
    )

    body_parts = [
        back_button,
        "<header>",
        f"  <h1>Benchmark Sample Report: <code>{html.escape(item.id)}</code></h1>",
        f"  <p class='card-subtitle'>Pipeline: <code>{html.escape(resp.pipeline_name)}</code> | Latency: <strong>{resp.latency_ms:.1f} ms</strong> | Reranker: <code>{html.escape(reranker_name)}</code></p>",
        "</header>",
        "<h2>1. Query Identification &amp; Metadata</h2>",
        "<div class='card' style='margin-bottom: 2rem;'>",
        "  <div class='meta-list'>",
        f"    <div class='meta-item'><span class='meta-label'>Sample ID:</span> <code>{html.escape(item.id)}</code></div>",
        f"    <div class='meta-item'><span class='meta-label'>Pipeline Architecture:</span> <code>{html.escape(resp.pipeline_name)}</code></div>",
        f"    <div class='meta-item'><span class='meta-label'>Investigative Typology:</span> <span class='badge badge-primary'>{html.escape(item.typology)}</span> &mdash; {typ_desc}</div>",
        f"    <div class='meta-item'><span class='meta-label'>Complexity Level:</span> <span class='badge badge-neutral'>{html.escape(item.complexity)}</span> &mdash; {comp_desc}</div>",
        f"    <div class='meta-item'><span class='meta-label'>Ground Truth Target Inquiries:</span> <code>{ref_str}</code></div>",
        f"    <div class='meta-item'><span class='meta-label'>Retrieved Inquiries (Retriever):</span> <code>{ret_str}</code></div>",
        f"    <div class='meta-item'><span class='meta-label'>Cited Inquiries (Generator):</span> <code>{cited_str}</code></div>",
        f"    <div class='meta-item'><span class='meta-label'>End-to-End Latency:</span> <strong>{resp.latency_ms:.1f} ms</strong></div>",
        f"    <div class='meta-item'><span class='meta-label'>Reranker Engine:</span> <code>{html.escape(reranker_name)}</code></div>",
        "  </div>",
        "</div>",
        "<h2>2. Evaluation Metrics Scorecard</h2>",
        "<div class='table-wrapper'>",
        "  <table>",
        "    <thead><tr><th>Metric Category</th><th>Metric</th><th class='num'>Score</th><th>Description</th></tr></thead>",
        "    <tbody>",
        f"      <tr><td>Retrieval</td><td>Precision@K</td><td class='num'>{ret.precision_at_k:.4f}</td><td>Proportion of retrieved chunks in top-K belonging to relevant inquiries</td></tr>",
        f"      <tr><td>Retrieval</td><td>Recall@K</td><td class='num'>{ret.recall_at_k:.4f}</td><td>Coverage of target ground truth inquiry references</td></tr>",
        f"      <tr><td>Retrieval</td><td>Hit Rate@K</td><td class='num'>{ret.hit_rate:.4f}</td><td>At least one relevant inquiry document identified</td></tr>",
        f"      <tr><td>Retrieval</td><td>Mean Reciprocal Rank (MRR)</td><td class='num'>{ret.mrr:.4f}</td><td>Reciprocal rank of the first relevant chunk</td></tr>",
        (
            f"<tr><td>Citation Attribution</td><td>Citation Precision</td><td class='num'>{cit.precision:.4f}</td><td>Proportion of generator-cited inquiries present in ground truth</td></tr>"
            if cit
            else "<tr><td>Citation Attribution</td><td>Citation Precision</td><td class='num'>N/A</td><td>Generator citation precision</td></tr>"
        ),
        (
            f"<tr><td>Citation Attribution</td><td>Citation Recall</td><td class='num'>{cit.recall:.4f}</td><td>Coverage of target ground truth references cited by generator</td></tr>"
            if cit
            else "<tr><td>Citation Attribution</td><td>Citation Recall</td><td class='num'>N/A</td><td>Generator citation recall</td></tr>"
        ),
        (
            f"<tr><td>Citation Attribution</td><td>Citation Hallucination Rate</td><td class='num'>{cit.hallucination_rate:.4f}</td><td>Proportion of cited inquiries not grounded in retrieved context</td></tr>"
            if cit
            else "<tr><td>Citation Attribution</td><td>Citation Hallucination Rate</td><td class='num'>N/A</td><td>Generator citation hallucination rate</td></tr>"
        ),
        f"      <tr><td>Lexical Overlap</td><td>ROUGE-L</td><td class='num'>{lex.rouge_l:.4f}</td><td>Longest common subsequence F1 against reference answer</td></tr>",
        f"      <tr><td>Lexical Overlap</td><td>Token F1</td><td class='num'>{lex.token_f1:.4f}</td><td>Harmonic mean of token precision and recall</td></tr>",
        f"      <tr><td>Lexical Overlap</td><td>BLEU-4</td><td class='num'>{lex.bleu_4:.4f}</td><td>4-gram precision against ground truth reference</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Faithfulness (Groundedness)</td><td class='num'>{judge.faithfulness:.4f}</td><td>Factual claims grounded strictly in retrieved context</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Answer Relevance</td><td class='num'>{judge.answer_relevance:.4f}</td><td>Direct alignment with the investigative query intent</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Context Recall</td><td class='num'>{judge.context_recall:.4f}</td><td>Presence of key investigative facts from reference answer in retrieved context</td></tr>",
        f"      <tr><td>LLM-as-a-Judge</td><td>Answer Correctness</td><td class='num'>{ans_corr:.4f}</td><td>Presence of key investigative facts from reference answer in generated answer</td></tr>",
        "    </tbody>",
        "  </table>",
        "</div>",
    ]

    if judge.critique:
        body_parts.extend(
            [
                "<div class='callout'>",
                "  <p><strong>LLM Judge Critique &amp; Justification:</strong></p>",
                f"  <p>{html.escape(judge.critique)}</p>",
                "</div>",
            ]
        )

    body_parts.extend(
        [
            "<h2>3. Investigative Query &amp; Complete Answers</h2>",
            "<h3>3.1 Investigative Question</h3>",
            f"<div class='answer-box'><strong>{html.escape(item.question)}</strong></div>",
            "<h3>3.2 RAG Generated Answer (Full)</h3>",
            f"<div class='answer-box'>{html.escape(resp.answer)}</div>",
            "<h3>3.3 Ground Truth Reference Answer (Full)</h3>",
            f"<div class='answer-box reference'>{html.escape(item.reference_answer)}</div>",
            "<h2>4. Executed Prompts</h2>",
            "<h3>4.1 RAG Generator &mdash; System Prompt</h3>",
            f"<pre><code>{html.escape(resp.system_prompt or '*No system prompt recorded.*')}</code></pre>",
            "<h3>4.2 RAG Generator &mdash; User Input Prompt (with Injected Context)</h3>",
            f"<pre><code>{html.escape(resp.prompt or '*No input prompt recorded.*')}</code></pre>",
            "<h3>4.3 LLM Judge &mdash; System Prompt</h3>",
            f"<pre><code>{html.escape(judge.system_prompt or '*No judge system prompt recorded.*')}</code></pre>",
            "<h3>4.4 LLM Judge &mdash; Evaluation Prompt</h3>",
            f"<pre><code>{html.escape(judge.prompt or '*No judge evaluation prompt recorded.*')}</code></pre>",
            "<h2>5. Retrieval &amp; Chunks Breakdown</h2>",
        ]
    )

    has_reranker = bool(resp.pre_rerank_chunks) and (resp.metadata.get("reranker") is not None)

    if has_reranker:
        pre_chunks = resp.pre_rerank_chunks
        final_chunks = resp.retrieved_chunks

        final_chunk_info: dict[str, tuple[int, float]] = {
            res.chunk.id: (rank, res.score) for rank, res in enumerate(final_chunks, start=1)
        }

        body_parts.extend(
            [
                "<h3>5.1 Reranking Transition &amp; Candidate Pool Overview</h3>",
                "<div class='table-wrapper'>",
                "  <table>",
                "    <thead><tr><th>Final Rank</th><th>Initial Rank</th><th>Shift (&Delta;)</th><th>Document ID</th><th>Chunk ID</th><th>Page</th><th class='num'>Initial Score</th><th class='num'>Rerank Score</th><th>Status</th></tr></thead>",
                "    <tbody>",
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
                status = "<span class='badge badge-success'>Promoted to Context</span>"
                rerank_score_str = f"{rerank_score:.4f}"
                final_rank_str = f"#{f_rank}"
            else:
                final_rank_str = "<em>Dropped</em>"
                delta_str = "&mdash;"
                status = "<span class='badge badge-danger'>Dropped from Context</span>"
                rerank_score_str = "&mdash;"

            body_parts.append(
                f"<tr>"
                f"<td><strong>{final_rank_str}</strong></td>"
                f"<td>#{init_rank}</td>"
                f"<td>{delta_str}</td>"
                f"<td><code>{html.escape(str(doc_id))}</code></td>"
                f"<td><code>{html.escape(str(cid))}</code></td>"
                f"<td>{html.escape(str(page))}</td>"
                f"<td class='num'>{init_score:.4f}</td>"
                f"<td class='num'>{rerank_score_str}</td>"
                f"<td>{status}</td>"
                f"</tr>"
            )

        body_parts.extend(
            [
                "    </tbody>",
                "  </table>",
                "</div>",
                "<h3>5.2 Final Chunks Promoted to Context</h3>",
            ]
        )

        for rank, res in enumerate(final_chunks, start=1):
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            init_rank = res.chunk.metadata.get("initial_retrieval_rank", "N/A")
            body_parts.extend(
                [
                    "<div class='card' style='margin-bottom: 1rem;'>",
                    f"  <h4 style='margin-top:0;'>Final Chunk #{rank} &mdash; Document: <code>{html.escape(str(doc_id))}</code> (Page {html.escape(str(page))})</h4>",
                    f"  <p style='font-size:0.85rem; color:var(--text-muted); margin-bottom:0.5rem;'>Chunk ID: <code>{html.escape(res.chunk.id)}</code> | Rerank Score: <strong>{res.score:.4f}</strong> | Initial Rank: <code>#{html.escape(str(init_rank))}</code></p>",
                    f"  <pre><code>{html.escape(res.chunk.text)}</code></pre>",
                    "</div>",
                ]
            )

        dropped_candidates = [
            (init_rank, res)
            for init_rank, res in enumerate(pre_chunks, start=1)
            if res.chunk.id not in final_chunk_info
        ]

        if dropped_candidates:
            body_parts.extend(
                [
                    "<h3>5.3 Dropped Candidate Chunks (Pre-Rerank Pool)</h3>",
                    f"<p><em>A total of {len(dropped_candidates)} candidate chunks were evaluated during initial retrieval but excluded from the final context after reranking.</em></p>",
                ]
            )
            for init_rank, res in dropped_candidates:
                doc_id = res.chunk.document_id
                page = res.chunk.metadata.get("page_number", "N/A")
                body_parts.extend(
                    [
                        "<details>",
                        f"  <summary>Candidate #{init_rank} (Dropped) &mdash; Document: <code>{html.escape(str(doc_id))}</code> (Page {html.escape(str(page))}) | Initial Score: {res.score:.4f} | Chunk ID: <code>{html.escape(res.chunk.id)}</code></summary>",
                        "  <div class='details-content'>",
                        f"    <pre><code>{html.escape(res.chunk.text)}</code></pre>",
                        "  </div>",
                        "</details>",
                    ]
                )
    else:
        body_parts.extend(
            [
                "<div class='callout warning'>",
                "  <p><strong>Reranker was not configured for this pipeline.</strong> Chunks below were retrieved directly via Top-K retrieval and forwarded to the generator context.</p>",
                "</div>",
                "<h3>5.1 Retrieved Context Chunks</h3>",
                "<div class='table-wrapper'>",
                "  <table>",
                "    <thead><tr><th>Rank</th><th>Document ID</th><th>Chunk ID</th><th>Page</th><th class='num'>Retrieval Score</th></tr></thead>",
                "    <tbody>",
            ]
        )
        for rank, res in enumerate(resp.retrieved_chunks, start=1):
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            body_parts.append(
                f"<tr>"
                f"<td>#{rank}</td>"
                f"<td><code>{html.escape(str(doc_id))}</code></td>"
                f"<td><code>{html.escape(str(res.chunk.id))}</code></td>"
                f"<td>{html.escape(str(page))}</td>"
                f"<td class='num'>{res.score:.4f}</td>"
                f"</tr>"
            )

        body_parts.extend(
            [
                "    </tbody>",
                "  </table>",
                "</div>",
                "<h3>5.2 Context Evidence Chunks (Fed to Generator)</h3>",
            ]
        )
        for rank, res in enumerate(resp.retrieved_chunks, start=1):
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            body_parts.extend(
                [
                    "<div class='card' style='margin-bottom: 1rem;'>",
                    f"  <h4 style='margin-top:0;'>Chunk #{rank} &mdash; Document: <code>{html.escape(str(doc_id))}</code> (Page {html.escape(str(page))})</h4>",
                    f"  <p style='font-size:0.85rem; color:var(--text-muted); margin-bottom:0.5rem;'>Chunk ID: <code>{html.escape(res.chunk.id)}</code> | Retrieval Score: <strong>{res.score:.4f}</strong></p>",
                    f"  <pre><code>{html.escape(res.chunk.text)}</code></pre>",
                    "</div>",
                ]
            )

    content = _html_page(title=f"Sample Report: {item.id}", body="\n".join(body_parts))
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")

    return content


def generate_all_question_reports(
    summary: EvaluationSummary,
    output_dir: Path | str,
    main_report_href: str | None = "../evaluation_report.html",
) -> list[Path]:
    """Generate detailed individual HTML reports for each benchmark question."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for s in summary.samples:
        safe_id = re.sub(r"[^\w\-]", "_", s.item.id)
        report_file = out_dir / f"{safe_id}.html"
        generate_question_report(s, output_path=report_file, main_report_href=main_report_href)
        generated.append(report_file)

    return generated


def generate_comparison_report(
    summary_a: EvaluationSummary,
    summary_b: EvaluationSummary,
    output_path: Path | str | None = None,
    report_a_href: str | None = None,
    report_b_href: str | None = None,
) -> str:
    """Generate a comparative HTML benchmark report comparing two RAG pipelines."""
    ov_a = summary_a.overall
    ov_b = summary_b.overall
    name_a = html.escape(summary_a.pipeline_name)
    name_b = html.escape(summary_b.pipeline_name)
    ac_a = getattr(ov_a, "mean_answer_correctness", 0.0)
    ac_b = getattr(ov_b, "mean_answer_correctness", 0.0)

    def diff_badge(val_b: float, val_a: float, lower_is_better: bool = False) -> str:
        d = val_b - val_a
        sign = "+" if d >= 0 else ""
        if abs(d) < 1e-9:
            badge_cls = "badge-neutral"
        elif lower_is_better:
            badge_cls = "badge-success" if d < 0 else "badge-danger"
        else:
            badge_cls = "badge-success" if d > 0 else "badge-danger"
        return f"<span class='badge {badge_cls}'>{sign}{d:.4f}</span>"

    links_html = ""
    if report_a_href or report_b_href:
        links_parts = ["<div class='card-grid' style='margin-bottom: 2rem;'>"]
        if report_a_href:
            links_parts.append(
                f"<div class='card'><div class='card-title'>Architecture A</div><div style='margin-top:0.25rem;'><a href='{html.escape(report_a_href)}' class='back-link' style='margin:0;'>View {name_a} Report &rarr;</a></div></div>"
            )
        if report_b_href:
            links_parts.append(
                f"<div class='card'><div class='card-title'>Architecture B</div><div style='margin-top:0.25rem;'><a href='{html.escape(report_b_href)}' class='back-link' style='margin:0;'>View {name_b} Report &rarr;</a></div></div>"
            )
        links_parts.append("</div>")
        links_html = "\n".join(links_parts)

    body_parts = [
        "<header>",
        f"  <h1>RAG Benchmark Architecture Comparison: <code>{name_a}</code> vs. <code>{name_b}</code></h1>",
        "  <p class='card-subtitle'>Direct head-to-head performance comparison, absolute metric deltas, and breakdown by investigative taxonomies.</p>",
        "</header>",
        links_html,
        "<h2>1. Overall Head-to-Head Comparison</h2>",
        "<div class='table-wrapper'>",
        "  <table>",
        f"    <thead><tr><th>Metric</th><th class='num'>{name_a} (A)</th><th class='num'>{name_b} (B)</th><th class='num'>Absolute Delta (B &minus; A)</th></tr></thead>",
        "    <tbody>",
        f"      <tr><td><strong>Retrieval Precision@K</strong></td><td class='num'>{ov_a.mean_precision_at_k:.4f}</td><td class='num'>{ov_b.mean_precision_at_k:.4f}</td><td class='num'>{diff_badge(ov_b.mean_precision_at_k, ov_a.mean_precision_at_k)}</td></tr>",
        f"      <tr><td><strong>Retrieval Recall@K</strong></td><td class='num'>{ov_a.mean_recall_at_k:.4f}</td><td class='num'>{ov_b.mean_recall_at_k:.4f}</td><td class='num'>{diff_badge(ov_b.mean_recall_at_k, ov_a.mean_recall_at_k)}</td></tr>",
        f"      <tr><td><strong>Retrieval Hit Rate</strong></td><td class='num'>{ov_a.mean_hit_rate:.4f}</td><td class='num'>{ov_b.mean_hit_rate:.4f}</td><td class='num'>{diff_badge(ov_b.mean_hit_rate, ov_a.mean_hit_rate)}</td></tr>",
        f"      <tr><td><strong>Retrieval MRR</strong></td><td class='num'>{ov_a.mean_mrr:.4f}</td><td class='num'>{ov_b.mean_mrr:.4f}</td><td class='num'>{diff_badge(ov_b.mean_mrr, ov_a.mean_mrr)}</td></tr>",
        f"      <tr><td><strong>Citation Precision</strong></td><td class='num'>{ov_a.mean_citation_precision:.4f}</td><td class='num'>{ov_b.mean_citation_precision:.4f}</td><td class='num'>{diff_badge(ov_b.mean_citation_precision, ov_a.mean_citation_precision)}</td></tr>",
        f"      <tr><td><strong>Citation Recall</strong></td><td class='num'>{ov_a.mean_citation_recall:.4f}</td><td class='num'>{ov_b.mean_citation_recall:.4f}</td><td class='num'>{diff_badge(ov_b.mean_citation_recall, ov_a.mean_citation_recall)}</td></tr>",
        f"      <tr><td><strong>Citation Hallucination Rate</strong></td><td class='num'>{ov_a.mean_citation_hallucination_rate:.4f}</td><td class='num'>{ov_b.mean_citation_hallucination_rate:.4f}</td><td class='num'>{diff_badge(ov_b.mean_citation_hallucination_rate, ov_a.mean_citation_hallucination_rate, lower_is_better=True)}</td></tr>",
        f"      <tr><td><strong>ROUGE-L</strong></td><td class='num'>{ov_a.mean_rouge_l:.4f}</td><td class='num'>{ov_b.mean_rouge_l:.4f}</td><td class='num'>{diff_badge(ov_b.mean_rouge_l, ov_a.mean_rouge_l)}</td></tr>",
        f"      <tr><td><strong>Token F1</strong></td><td class='num'>{ov_a.mean_token_f1:.4f}</td><td class='num'>{ov_b.mean_token_f1:.4f}</td><td class='num'>{diff_badge(ov_b.mean_token_f1, ov_a.mean_token_f1)}</td></tr>",
        f"      <tr><td><strong>BLEU-4</strong></td><td class='num'>{ov_a.mean_bleu_4:.4f}</td><td class='num'>{ov_b.mean_bleu_4:.4f}</td><td class='num'>{diff_badge(ov_b.mean_bleu_4, ov_a.mean_bleu_4)}</td></tr>",
        f"      <tr><td><strong>Faithfulness (Judge)</strong></td><td class='num'>{ov_a.mean_faithfulness:.4f}</td><td class='num'>{ov_b.mean_faithfulness:.4f}</td><td class='num'>{diff_badge(ov_b.mean_faithfulness, ov_a.mean_faithfulness)}</td></tr>",
        f"      <tr><td><strong>Relevance (Judge)</strong></td><td class='num'>{ov_a.mean_answer_relevance:.4f}</td><td class='num'>{ov_b.mean_answer_relevance:.4f}</td><td class='num'>{diff_badge(ov_b.mean_answer_relevance, ov_a.mean_answer_relevance)}</td></tr>",
        f"      <tr><td><strong>Context Recall (Judge)</strong></td><td class='num'>{ov_a.mean_context_recall:.4f}</td><td class='num'>{ov_b.mean_context_recall:.4f}</td><td class='num'>{diff_badge(ov_b.mean_context_recall, ov_a.mean_context_recall)}</td></tr>",
        f"      <tr><td><strong>Answer Correctness (Judge)</strong></td><td class='num'>{ac_a:.4f}</td><td class='num'>{ac_b:.4f}</td><td class='num'>{diff_badge(ac_b, ac_a)}</td></tr>",
        f"      <tr><td><strong>Mean Latency (ms)</strong></td><td class='num'>{ov_a.mean_latency_ms:.1f}</td><td class='num'>{ov_b.mean_latency_ms:.1f}</td><td class='num'>{diff_badge(ov_b.mean_latency_ms, ov_a.mean_latency_ms, lower_is_better=True)}</td></tr>",
        "    </tbody>",
        "  </table>",
        "</div>",
        _render_metric_guide_html(),
        "<h2>2. Pipeline Architectures &amp; Models Comparison</h2>",
        _render_comparison_config_html(summary_a, summary_b),
        "<h2>3. Typology Comparison (Recall@K / Faithfulness / Context Recall / Answer Correctness)</h2>",
        "<div class='table-wrapper'>",
        "  <table>",
        f"    <thead><tr><th>Typology</th><th class='num'>{name_a} Recall</th><th class='num'>{name_b} Recall</th><th class='num'>{name_a} Faith</th><th class='num'>{name_b} Faith</th><th class='num'>{name_a} CtxRec</th><th class='num'>{name_b} CtxRec</th><th class='num'>{name_a} AnsCorr</th><th class='num'>{name_b} AnsCorr</th></tr></thead>",
        "    <tbody>",
    ]

    all_typs = sorted(set(summary_a.by_typology.keys()).union(set(summary_b.by_typology.keys())))
    for typ in all_typs:
        ta = summary_a.by_typology.get(typ)
        tb = summary_b.by_typology.get(typ)
        rec_a = f"{ta.mean_recall_at_k:.4f}" if ta else "N/A"
        rec_b = f"{tb.mean_recall_at_k:.4f}" if tb else "N/A"
        f_a = f"{ta.mean_faithfulness:.4f}" if ta else "N/A"
        f_b = f"{tb.mean_faithfulness:.4f}" if tb else "N/A"
        cr_a = f"{ta.mean_context_recall:.4f}" if ta else "N/A"
        cr_b = f"{tb.mean_context_recall:.4f}" if tb else "N/A"
        ac_ta = f"{getattr(ta, 'mean_answer_correctness', 0.0):.4f}" if ta else "N/A"
        ac_tb = f"{getattr(tb, 'mean_answer_correctness', 0.0):.4f}" if tb else "N/A"
        body_parts.append(
            f"<tr><td><strong>{html.escape(typ)}</strong></td><td class='num'>{rec_a}</td><td class='num'>{rec_b}</td><td class='num'>{f_a}</td><td class='num'>{f_b}</td><td class='num'>{cr_a}</td><td class='num'>{cr_b}</td><td class='num'>{ac_ta}</td><td class='num'>{ac_tb}</td></tr>"
        )

    body_parts.extend(
        [
            "    </tbody>",
            "  </table>",
            "</div>",
            "<h2>4. Complexity Comparison (Recall@K / Faithfulness / Context Recall / Answer Correctness)</h2>",
            "<div class='table-wrapper'>",
            "  <table>",
            f"    <thead><tr><th>Complexity</th><th class='num'>{name_a} Recall</th><th class='num'>{name_b} Recall</th><th class='num'>{name_a} Faith</th><th class='num'>{name_b} Faith</th><th class='num'>{name_a} CtxRec</th><th class='num'>{name_b} CtxRec</th><th class='num'>{name_a} AnsCorr</th><th class='num'>{name_b} AnsCorr</th></tr></thead>",
            "    <tbody>",
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
        cr_a = f"{ca.mean_context_recall:.4f}" if ca else "N/A"
        cr_b = f"{cb.mean_context_recall:.4f}" if cb else "N/A"
        ac_ca = f"{getattr(ca, 'mean_answer_correctness', 0.0):.4f}" if ca else "N/A"
        ac_cb = f"{getattr(cb, 'mean_answer_correctness', 0.0):.4f}" if cb else "N/A"
        body_parts.append(
            f"<tr><td><strong>{html.escape(comp)}</strong></td><td class='num'>{rec_a}</td><td class='num'>{rec_b}</td><td class='num'>{f_a}</td><td class='num'>{f_b}</td><td class='num'>{cr_a}</td><td class='num'>{cr_b}</td><td class='num'>{ac_ca}</td><td class='num'>{ac_cb}</td></tr>"
        )

    body_parts.extend(["    </tbody>", "  </table>", "</div>"])

    content = _html_page(
        title=f"Comparison: {summary_a.pipeline_name} vs {summary_b.pipeline_name}",
        body="\n".join(body_parts),
    )
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(content, encoding="utf-8")

    return content
