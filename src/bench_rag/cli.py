"""Command-line interface for the Modular RAG Evaluation Framework."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from bench_rag.chunkers import get_chunker
from bench_rag.config import PipelineConfig, load_pipeline_config
from bench_rag.core.chunker import Chunk
from bench_rag.core.embedding import EmbeddingModel
from bench_rag.core.parser import Document, DocumentParser
from bench_rag.core.vector_store import VectorStore
from bench_rag.embeddings import get_embedding_model
from bench_rag.evaluation.engine import StandardEvaluationEngine, load_ground_truth_csv
from bench_rag.parsers import get_parser
from bench_rag.pipelines import build_pipeline_from_config
from bench_rag.reporting import (
    export_document_chunks,
    export_evaluation_json,
    export_samples_csv,
    format_file_size,
    generate_all_question_reports,
    generate_comparison_report,
    generate_markdown_report,
    track_embeddings,
    track_files,
    track_indexing,
    track_queries,
)
from bench_rag.vector_stores import get_vector_store

app = typer.Typer(
    name="bench-rag",
    help="Modular RAG Evaluation Framework for Criminal Investigation Reports.",
    add_completion=False,
)
console = Console()


def _build_parser_from_config(
    cfg: PipelineConfig, clean_override: bool | None = None
) -> DocumentParser:
    """Build a DocumentParser respecting PipelineConfig and optional CLI flag override."""
    should_clean = clean_override if clean_override is not None else cfg.clean_headers_footers
    return get_parser(
        parser_type=cfg.parser_type,
        clean_headers_footers=should_clean,
        header_margin=cfg.header_margin,
        footer_margin=cfg.footer_margin,
        detect_repeated_headers=cfg.detect_repeated_headers,
        repeated_header_threshold=cfg.repeated_header_threshold,
        remove_pje_metadata=cfg.remove_pje_metadata,
        remove_page_numbers=cfg.remove_page_numbers,
        preserve_first_header=cfg.preserve_first_header,
        custom_header_patterns=cfg.custom_header_patterns,
        custom_footer_patterns=cfg.custom_footer_patterns,
    )


@app.command("seed-sample-data")
def seed_sample_data(
    data_dir: str = typer.Option("data", help="Target base directory for sample data."),
) -> None:
    """Generate synthetic forensic investigation reports (PDF) and ground truth benchmark CSV."""
    from bench_rag.data_gen.seed import generate_sample_dataset

    with console.status(
        "[bold green]Generating synthetic forensic reports and ground truth dataset..."
    ):
        generate_sample_dataset(data_dir=data_dir)

    console.print(
        f"[bold green]✓[/bold green] Sample forensic reports generated in: [bold cyan]{data_dir}/raw_reports[/bold cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] Ground truth benchmark dataset generated in: [bold cyan]{data_dir}/ground_truth.csv[/bold cyan]"
    )


@app.command("ingest")
def ingest(
    input_dir: str = typer.Option(
        "data/raw_reports", help="Directory containing forensic PDF reports."
    ),
    parser_type: str = typer.Option("pymupdf", help="Parser engine ('pymupdf' or 'pdfplumber')."),
    clean_headers_footers: bool = typer.Option(
        False,
        "--clean-headers-footers/--no-clean-headers-footers",
        help="Remove repeated headers, footers, and PJe metadata prior to chunking.",
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable interactive progress bars."
    ),
) -> None:
    """Parse forensic PDF reports from a directory and display extracted document metadata."""
    parser = get_parser(parser_type=parser_type, clean_headers_footers=clean_headers_footers)
    pdf_files = sorted(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        console.print(f"[bold red]Error:[/bold red] No PDF reports found in {input_dir}")
        raise typer.Exit(code=1)

    table = Table(title=f"Ingested Forensic Documents ({parser_type})")
    table.add_column("Doc ID (Normalized)", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")
    table.add_column("Pages", justify="right", style="green")
    if clean_headers_footers:
        table.add_column("Cleaned", justify="center", style="yellow")
        table.add_column("Chars Removed", justify="right", style="red")
    table.add_column("Total Characters", justify="right")

    with track_files(
        pdf_files,
        description="[bold cyan]Ingesting PDF reports[/bold cyan]",
        no_progress=no_progress,
    ) as tracker:
        for pdf_path in pdf_files:
            size_str = format_file_size(pdf_path.stat().st_size)
            tracker.update_detail(f"File: {pdf_path.name} ({size_str})")
            docs = parser.parse(pdf_path)
            for doc in docs:
                row = [
                    doc.id,
                    doc.metadata.get("source_file", pdf_path.name),
                    str(doc.metadata.get("total_pages", "N/A")),
                ]
                if clean_headers_footers:
                    row.append("Yes" if doc.metadata.get("cleaned_headers_footers") else "No")
                    row.append(f"{doc.metadata.get('characters_removed', 0):,}")
                row.append(f"{len(doc.text):,}")
                table.add_row(*row)
            tracker.advance(1)

    console.print(table)


def _index_chunks_with_progress(
    chunks: list[Chunk],
    embedding_model: EmbeddingModel,
    vector_store: VectorStore,
    collection_name: str,
    vector_store_type: str,
    no_progress: bool = False,
) -> list[list[float]]:
    """Generate embeddings and index chunks with multi-line, batch-level progress reporting."""
    if not chunks:
        return []

    # 1. Embedding generation stage
    with track_embeddings(
        len(chunks),
        description=f"[bold magenta]Generating embeddings for {len(chunks)} chunks[/bold magenta]",
        no_progress=no_progress,
    ) as tracker:

        def on_embed_progress(completed: int, total: int) -> None:
            tracker.update_progress(completed)
            tracker.update_detail(f"Embedded {completed}/{total} chunks...")

        texts = [c.text for c in chunks]
        embeddings = embedding_model.embed_texts(texts, progress_callback=on_embed_progress)
        tracker.update_detail("Embedding generation completed.")

    # 2. Vector store indexing stage
    with track_indexing(
        len(chunks),
        description=f"[bold blue]Indexing into {vector_store_type}[/bold blue]",
        no_progress=no_progress,
    ) as tracker:
        tracker.update_detail(f"Configuring collection '{collection_name}'...")
        vector_store.create_collection(
            collection_name=collection_name,
            dimension=embedding_model.dimension,
        )

        def on_index_progress(completed: int, total: int) -> None:
            tracker.update_progress(completed)
            tracker.update_detail(f"Indexed {completed}/{total} chunks into '{collection_name}'...")

        vector_store.add_chunks(
            collection_name=collection_name,
            chunks=chunks,
            embeddings=embeddings,
            progress_callback=on_index_progress,
        )
        tracker.update_detail("Indexing completed.")

    return embeddings


@app.command("index")
def index(
    input_dir: str = typer.Option(
        "data/raw_reports", help="Directory containing forensic PDF reports."
    ),
    pipeline_config: str | None = typer.Option(
        None, help="Path to YAML pipeline configuration file."
    ),
    clean_headers_footers: bool | None = typer.Option(
        None,
        "--clean-headers-footers/--no-clean-headers-footers",
        help="Override pipeline config to remove repeated headers, footers, and PJe metadata.",
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable interactive progress bars."
    ),
) -> None:
    """Ingest, chunk, embed, and index forensic documents into the vector store."""
    cfg = load_pipeline_config(pipeline_config)
    parser = _build_parser_from_config(cfg, clean_override=clean_headers_footers)
    chunker = get_chunker(
        chunker_type=cfg.chunker_type,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    embedding_model = get_embedding_model(
        embedding_type=cfg.embedding_type,
        model_name=cfg.embedding_model,
        base_url=cfg.embedding_base_url,
        query_prefix=cfg.embedding_query_prefix,
        document_prefix=cfg.embedding_document_prefix,
    )
    vector_store = get_vector_store(store_type=cfg.vector_store_type)

    pdf_files = sorted(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        console.print(f"[bold red]Error:[/bold red] No PDF reports found in {input_dir}")
        raise typer.Exit(code=1)

    # Ingest
    all_docs: list[Document] = []
    with track_files(
        pdf_files,
        description="[bold cyan]Parsing PDF reports[/bold cyan]",
        no_progress=no_progress,
    ) as tracker:
        for pdf_p in pdf_files:
            size_str = format_file_size(pdf_p.stat().st_size)
            tracker.update_detail(f"File: {pdf_p.name} ({size_str})")
            all_docs.extend(parser.parse(pdf_p))
            tracker.advance(1)

    # Chunk
    chunks = chunker.chunk(all_docs)

    # Embed & Index
    _index_chunks_with_progress(
        chunks=chunks,
        embedding_model=embedding_model,
        vector_store=vector_store,
        collection_name=cfg.collection_name,
        vector_store_type=cfg.vector_store_type,
        no_progress=no_progress,
    )

    console.print(
        f"[bold green]✓[/bold green] Indexed [bold cyan]{len(chunks)}[/bold cyan] chunks from "
        f"[bold cyan]{len(all_docs)}[/bold cyan] documents into collection [bold yellow]{cfg.collection_name}[/bold yellow]!"
    )


@app.command("evaluate")
def evaluate(
    benchmark_csv: str = typer.Option(
        "data/ground_truth.csv", help="Path to ground truth benchmark CSV."
    ),
    pipeline_config: str | None = typer.Option(
        None, help="Path to YAML pipeline configuration file."
    ),
    input_dir: str = typer.Option(
        "data/raw_reports", help="Directory containing forensic PDF reports."
    ),
    output_dir: str = typer.Option(
        "reports/run_default", help="Directory to save evaluation reports."
    ),
    top_k: int | None = typer.Option(
        None, help="Number of retrieved chunks per query (overrides pipeline config)."
    ),
    clean_headers_footers: bool | None = typer.Option(
        None,
        "--clean-headers-footers/--no-clean-headers-footers",
        help="Override pipeline config to remove repeated headers, footers, and PJe metadata.",
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable interactive progress bars."
    ),
    skip_indexing: bool = typer.Option(
        False,
        "--skip-indexing",
        "--skip-ingest",
        help="Skip document ingestion and indexing; evaluate against an existing vector store index.",
    ),
) -> None:
    """Run benchmark evaluation on a RAG pipeline and export reports."""
    cfg = load_pipeline_config(pipeline_config)
    pipeline = build_pipeline_from_config(cfg)
    effective_top_k = top_k if top_k is not None else cfg.top_k

    all_docs: list[Document] = []
    chunks: list[Chunk] = []

    if not skip_indexing:
        # Ensure documents are indexed into the pipeline's vector store
        parser = _build_parser_from_config(cfg, clean_override=clean_headers_footers)
        chunker = get_chunker(
            chunker_type=cfg.chunker_type,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        pdf_files = sorted(Path(input_dir).glob("*.pdf"))
        if not pdf_files:
            console.print(
                f"[bold red]Error:[/bold red] No PDF reports found in '{input_dir}'. "
                f"If the vector database is already populated, run with [bold cyan]--skip-indexing[/bold cyan]."
            )
            raise typer.Exit(code=1)

        with track_files(
            pdf_files,
            description="[bold cyan]Ingesting PDF reports[/bold cyan]",
            no_progress=no_progress,
        ) as tracker:
            for pdf in pdf_files:
                size_str = format_file_size(pdf.stat().st_size)
                tracker.update_detail(f"File: {pdf.name} ({size_str})")
                all_docs.extend(parser.parse(pdf))
                tracker.advance(1)

        chunks = chunker.chunk(all_docs)
        _index_chunks_with_progress(
            chunks=chunks,
            embedding_model=pipeline.embedding_model,
            vector_store=pipeline.vector_store,
            collection_name=cfg.collection_name,
            vector_store_type=cfg.vector_store_type,
            no_progress=no_progress,
        )
    else:
        chunk_count = pipeline.vector_store.count_chunks(cfg.collection_name)
        if chunk_count == 0:
            console.print(
                f"[bold red]Error:[/bold red] Collection '[bold cyan]{cfg.collection_name}[/bold cyan]' "
                f"does not exist or has 0 chunks in vector store '{cfg.vector_store_type}'. "
                f"Make sure documents were indexed beforehand (e.g. via 'bench-rag index')."
            )
            raise typer.Exit(code=1)

        console.print(
            f"[bold green]✓[/bold green] Skipping ingestion and indexing. "
            f"Evaluating against existing collection '[bold yellow]{cfg.collection_name}[/bold yellow]' "
            f"([bold cyan]{chunk_count}[/bold cyan] chunks available)."
        )

    # Load dataset
    dataset = load_ground_truth_csv(benchmark_csv)

    # Run evaluation
    engine = StandardEvaluationEngine(
        judge_type=cfg.judge_type,
        judge_model=cfg.judge_model,
        judge_base_url=cfg.judge_base_url,
        judge_enable_thinking=cfg.judge_enable_thinking,
        judge_dialect=cfg.judge_dialect,
        judge_max_tokens=cfg.judge_max_tokens,
    )
    with track_queries(
        len(dataset),
        description=f"[bold green]Evaluating pipeline '{cfg.name}'[/bold green]",
        no_progress=no_progress,
    ) as tracker:
        summary = engine.evaluate(
            dataset=dataset,
            pipeline=pipeline,
            top_k=effective_top_k,
            progress_callback=tracker.evaluation_callback,
        )

    # Save reports
    out_p = Path(output_dir)
    md_file = out_p / "evaluation_report.md"
    json_file = out_p / "run_summary.json"
    csv_file = out_p / "samples_evaluation.csv"
    questions_dir = out_p / "questions"
    chunks_dir = out_p / "chunks"

    generate_markdown_report(summary, output_path=md_file, question_rel_dir="questions")
    export_evaluation_json(summary, output_path=json_file)
    export_samples_csv(summary, output_path=csv_file)
    generate_all_question_reports(summary, output_dir=questions_dir)
    if chunks:
        export_document_chunks(chunks=chunks, output_dir=chunks_dir, documents=all_docs)

    # Display Rich summary table
    ov = summary.overall
    t = Table(title=f"Evaluation Summary: {summary.pipeline_name}")
    t.add_column("Category", style="cyan")
    t.add_column("Recall@K", justify="right", style="green")
    t.add_column("MRR", justify="right", style="green")
    t.add_column("ROUGE-L", justify="right")
    t.add_column("Token F1", justify="right")
    t.add_column("Faithfulness", justify="right", style="yellow")
    t.add_column("Relevance", justify="right", style="yellow")
    t.add_column("Latency (ms)", justify="right")

    t.add_row(
        "OVERALL",
        f"{ov.mean_recall_at_k:.4f}",
        f"{ov.mean_mrr:.4f}",
        f"{ov.mean_rouge_l:.4f}",
        f"{ov.mean_token_f1:.4f}",
        f"{ov.mean_faithfulness:.4f}",
        f"{ov.mean_answer_relevance:.4f}",
        f"{ov.mean_latency_ms:.1f}",
    )

    for typ, agg in summary.by_typology.items():
        t.add_row(
            f"Typology {typ}",
            f"{agg.mean_recall_at_k:.4f}",
            f"{agg.mean_mrr:.4f}",
            f"{agg.mean_rouge_l:.4f}",
            f"{agg.mean_token_f1:.4f}",
            f"{agg.mean_faithfulness:.4f}",
            f"{agg.mean_answer_relevance:.4f}",
            f"{agg.mean_latency_ms:.1f}",
        )

    for comp, agg in summary.by_complexity.items():
        t.add_row(
            f"Complexity {comp}",
            f"{agg.mean_recall_at_k:.4f}",
            f"{agg.mean_mrr:.4f}",
            f"{agg.mean_rouge_l:.4f}",
            f"{agg.mean_token_f1:.4f}",
            f"{agg.mean_faithfulness:.4f}",
            f"{agg.mean_answer_relevance:.4f}",
            f"{agg.mean_latency_ms:.1f}",
        )

    console.print(t)
    console.print(
        f"[bold green]✓[/bold green] Markdown report written to: [bold cyan]{md_file}[/bold cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] JSON summary written to: [bold cyan]{json_file}[/bold cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] CSV samples written to: [bold cyan]{csv_file}[/bold cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] Individual question reports written to: [bold cyan]{questions_dir}[/bold cyan]"
    )
    if chunks:
        console.print(
            f"[bold green]✓[/bold green] Document chunks exported to: [bold cyan]{chunks_dir}[/bold cyan]"
        )


@app.command("compare")
def compare(
    benchmark_csv: str = typer.Option(
        "data/ground_truth.csv", help="Path to ground truth benchmark CSV."
    ),
    config_a: str = typer.Option(
        "configs/pipeline_default.yaml", help="Config for Pipeline A (Standard Dense RAG)."
    ),
    config_b: str = typer.Option(
        "configs/pipeline_hybrid.yaml", help="Config for Pipeline B (Hybrid RAG)."
    ),
    input_dir: str = typer.Option(
        "data/raw_reports", help="Directory containing forensic PDF reports."
    ),
    output_dir: str = typer.Option(
        "reports/comparison_run", help="Directory to save comparative reports."
    ),
    top_k: int | None = typer.Option(
        None, help="Number of retrieved chunks per query (overrides pipeline configs)."
    ),
    clean_headers_footers: bool | None = typer.Option(
        None,
        "--clean-headers-footers/--no-clean-headers-footers",
        help="Override pipeline config to remove repeated headers, footers, and PJe metadata.",
    ),
    no_progress: bool = typer.Option(
        False, "--no-progress", help="Disable interactive progress bars."
    ),
    skip_indexing: bool = typer.Option(
        False,
        "--skip-indexing",
        "--skip-ingest",
        help="Skip document ingestion and indexing; evaluate against existing vector store collections.",
    ),
) -> None:
    """Run comparative benchmarking between two RAG pipeline architectures."""
    cfg_a = load_pipeline_config(config_a)
    cfg_b = load_pipeline_config(config_b)
    top_k_a = top_k if top_k is not None else cfg_a.top_k
    top_k_b = top_k if top_k is not None else cfg_b.top_k

    pipe_a = build_pipeline_from_config(cfg_a)
    pipe_b = build_pipeline_from_config(cfg_b)

    dataset = load_ground_truth_csv(benchmark_csv)

    all_docs: list[Document] = []
    chunks_a: list[Chunk] = []
    chunks_b: list[Chunk] = []
    distinct_chunking = False

    if not skip_indexing:
        # Ingest PDFs
        parser = _build_parser_from_config(cfg_a, clean_override=clean_headers_footers)
        chunker_a = get_chunker(
            chunker_type=cfg_a.chunker_type,
            chunk_size=cfg_a.chunk_size,
            chunk_overlap=cfg_a.chunk_overlap,
        )
        pdf_files = sorted(Path(input_dir).glob("*.pdf"))
        if not pdf_files:
            console.print(
                f"[bold red]Error:[/bold red] No PDF reports found in '{input_dir}'. "
                f"If the vector database is already populated, run with [bold cyan]--skip-indexing[/bold cyan]."
            )
            raise typer.Exit(code=1)

        with track_files(
            pdf_files,
            description="[bold cyan]Ingesting PDF reports for comparison[/bold cyan]",
            no_progress=no_progress,
        ) as tracker:
            for pdf in pdf_files:
                size_str = format_file_size(pdf.stat().st_size)
                tracker.update_detail(f"File: {pdf.name} ({size_str})")
                all_docs.extend(parser.parse(pdf))
                tracker.advance(1)

        chunks_a = chunker_a.chunk(all_docs)
        texts_a = [c.text for c in chunks_a]

        distinct_chunking = (
            cfg_a.chunker_type != cfg_b.chunker_type
            or cfg_a.chunk_size != cfg_b.chunk_size
            or cfg_a.chunk_overlap != cfg_b.chunk_overlap
        )
        if distinct_chunking:
            chunker_b = get_chunker(
                chunker_type=cfg_b.chunker_type,
                chunk_size=cfg_b.chunk_size,
                chunk_overlap=cfg_b.chunk_overlap,
            )
            chunks_b = chunker_b.chunk(all_docs)
            texts_b = [c.text for c in chunks_b]
        else:
            chunks_b = chunks_a
            texts_b = texts_a

        # Index Pipeline A
        emb_a = pipe_a.embedding_model.embed_texts(texts_a)
        pipe_a.vector_store.create_collection(
            collection_name=cfg_a.collection_name, dimension=pipe_a.embedding_model.dimension
        )
        pipe_a.vector_store.add_chunks(
            collection_name=cfg_a.collection_name, chunks=chunks_a, embeddings=emb_a
        )

        # Index Pipeline B
        emb_b = pipe_b.embedding_model.embed_texts(texts_b)
        pipe_b.vector_store.create_collection(
            collection_name=cfg_b.collection_name, dimension=pipe_b.embedding_model.dimension
        )
        pipe_b.vector_store.add_chunks(
            collection_name=cfg_b.collection_name, chunks=chunks_b, embeddings=emb_b
        )
    else:
        count_a = pipe_a.vector_store.count_chunks(cfg_a.collection_name)
        if count_a == 0:
            console.print(
                f"[bold red]Error:[/bold red] Collection '[bold cyan]{cfg_a.collection_name}[/bold cyan]' "
                f"for Pipeline A does not exist or has 0 chunks in vector store '{cfg_a.vector_store_type}'. "
                f"Make sure documents were indexed beforehand (e.g. via 'bench-rag index')."
            )
            raise typer.Exit(code=1)

        count_b = pipe_b.vector_store.count_chunks(cfg_b.collection_name)
        if count_b == 0:
            console.print(
                f"[bold red]Error:[/bold red] Collection '[bold cyan]{cfg_b.collection_name}[/bold cyan]' "
                f"for Pipeline B does not exist or has 0 chunks in vector store '{cfg_b.vector_store_type}'. "
                f"Make sure documents were indexed beforehand (e.g. via 'bench-rag index')."
            )
            raise typer.Exit(code=1)

        console.print(
            f"[bold green]✓[/bold green] Skipping ingestion and indexing. "
            f"Evaluating against existing collections: '[bold yellow]{cfg_a.collection_name}[/bold yellow]' "
            f"([bold cyan]{count_a}[/bold cyan] chunks) and '[bold yellow]{cfg_b.collection_name}[/bold yellow]' "
            f"([bold cyan]{count_b}[/bold cyan] chunks)."
        )

    # Evaluate Pipeline A
    engine_a = StandardEvaluationEngine(
        judge_type=cfg_a.judge_type,
        judge_model=cfg_a.judge_model,
        judge_base_url=cfg_a.judge_base_url,
        judge_enable_thinking=cfg_a.judge_enable_thinking,
        judge_dialect=cfg_a.judge_dialect,
    )
    with track_queries(
        len(dataset),
        description=f"[bold cyan]Benchmarking Pipeline A ({cfg_a.name})[/bold cyan]",
        no_progress=no_progress,
    ) as tracker_a:
        summary_a = engine_a.evaluate(
            dataset=dataset,
            pipeline=pipe_a,
            top_k=top_k_a,
            progress_callback=tracker_a.evaluation_callback,
        )

    # Evaluate Pipeline B
    engine_b = StandardEvaluationEngine(
        judge_type=cfg_b.judge_type,
        judge_model=cfg_b.judge_model,
        judge_base_url=cfg_b.judge_base_url,
        judge_enable_thinking=cfg_b.judge_enable_thinking,
        judge_dialect=cfg_b.judge_dialect,
    )
    with track_queries(
        len(dataset),
        description=f"[bold magenta]Benchmarking Pipeline B ({cfg_b.name})[/bold magenta]",
        no_progress=no_progress,
    ) as tracker_b:
        summary_b = engine_b.evaluate(
            dataset=dataset,
            pipeline=pipe_b,
            top_k=top_k_b,
            progress_callback=tracker_b.evaluation_callback,
        )

    # Generate comparative reports
    out_p = Path(output_dir)
    comp_md = out_p / "comparison_report.md"
    q_dir_a = out_p / "questions" / summary_a.pipeline_name
    q_dir_b = out_p / "questions" / summary_b.pipeline_name

    generate_comparison_report(summary_a, summary_b, output_path=comp_md)
    generate_markdown_report(
        summary_a,
        output_path=out_p / f"{summary_a.pipeline_name}_report.md",
        question_rel_dir=f"questions/{summary_a.pipeline_name}",
    )
    generate_markdown_report(
        summary_b,
        output_path=out_p / f"{summary_b.pipeline_name}_report.md",
        question_rel_dir=f"questions/{summary_b.pipeline_name}",
    )
    generate_all_question_reports(summary_a, output_dir=q_dir_a)
    generate_all_question_reports(summary_b, output_dir=q_dir_b)

    chunks_export_msg = None
    if chunks_a:
        if distinct_chunking:
            chunks_dir_a = out_p / "chunks" / summary_a.pipeline_name
            chunks_dir_b = out_p / "chunks" / summary_b.pipeline_name
            export_document_chunks(chunks=chunks_a, output_dir=chunks_dir_a, documents=all_docs)
            export_document_chunks(chunks=chunks_b, output_dir=chunks_dir_b, documents=all_docs)
            chunks_export_msg = f"{out_p / 'chunks'} (subdiretórios por arquitetura)"
        else:
            chunks_dir = out_p / "chunks"
            export_document_chunks(chunks=chunks_a, output_dir=chunks_dir, documents=all_docs)
            chunks_export_msg = str(chunks_dir)

    # Display comparison table
    t = Table(title="Architecture Comparison: Standard Dense RAG vs. Hybrid RAG")
    t.add_column("Metric", style="cyan")
    t.add_column(summary_a.pipeline_name, justify="right")
    t.add_column(summary_b.pipeline_name, justify="right")
    t.add_column("Delta (B - A)", justify="right", style="green")

    metrics_list = [
        (
            "Precision@K",
            summary_a.overall.mean_precision_at_k,
            summary_b.overall.mean_precision_at_k,
        ),
        ("Recall@K", summary_a.overall.mean_recall_at_k, summary_b.overall.mean_recall_at_k),
        ("Hit Rate", summary_a.overall.mean_hit_rate, summary_b.overall.mean_hit_rate),
        ("MRR", summary_a.overall.mean_mrr, summary_b.overall.mean_mrr),
        ("ROUGE-L", summary_a.overall.mean_rouge_l, summary_b.overall.mean_rouge_l),
        ("Token F1", summary_a.overall.mean_token_f1, summary_b.overall.mean_token_f1),
        ("Faithfulness", summary_a.overall.mean_faithfulness, summary_b.overall.mean_faithfulness),
        (
            "Relevance",
            summary_a.overall.mean_answer_relevance,
            summary_b.overall.mean_answer_relevance,
        ),
        ("Latency (ms)", summary_a.overall.mean_latency_ms, summary_b.overall.mean_latency_ms),
    ]

    for name, val_a, val_b in metrics_list:
        delta = val_b - val_a
        delta_str = f"{'+' if delta >= 0 else ''}{delta:.4f}"
        t.add_row(name, f"{val_a:.4f}", f"{val_b:.4f}", delta_str)

    console.print(t)
    console.print(
        f"[bold green]✓[/bold green] Comparison Markdown report written to: [bold cyan]{comp_md}[/bold cyan]"
    )
    console.print(
        f"[bold green]✓[/bold green] Individual question reports written to: [bold cyan]{out_p / 'questions'}[/bold cyan]"
    )
    if chunks_export_msg:
        console.print(
            f"[bold green]✓[/bold green] Document chunks exported to: [bold cyan]{chunks_export_msg}[/bold cyan]"
        )


if __name__ == "__main__":
    app()
