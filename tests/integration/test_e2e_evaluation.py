"""End-to-end integration test of the benchmark framework."""

from pathlib import Path

from bench_rag.chunkers import RecursiveCharacterChunker
from bench_rag.core.chunker import Chunk
from bench_rag.data_gen.seed import generate_sample_dataset
from bench_rag.embeddings import MockEmbeddingModel
from bench_rag.evaluation.engine import StandardEvaluationEngine, load_ground_truth_csv
from bench_rag.generators import MockGeneratorLLM
from bench_rag.parsers import PyMuPDFParser
from bench_rag.pipelines import HybridRAGPipeline, StandardDenseRAGPipeline
from bench_rag.reporting import (
    export_document_chunks,
    export_evaluation_json,
    export_samples_csv,
    generate_all_question_reports,
    generate_comparison_report,
    generate_markdown_report,
)
from bench_rag.vector_stores import MemoryVectorStore


def test_full_pipeline_e2e(tmp_path: Path):
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"

    # 1. Seed dataset
    generate_sample_dataset(data_dir=data_dir)
    raw_dir = data_dir / "raw_reports"
    gt_csv = data_dir / "ground_truth.csv"

    assert len(list(raw_dir.glob("*.pdf"))) == 3
    assert gt_csv.exists()

    # 2. Ingest
    parser = PyMuPDFParser()
    docs = []
    for pdf_p in raw_dir.glob("*.pdf"):
        docs.extend(parser.parse(pdf_p))
    assert len(docs) == 3

    # 3. Chunk
    chunker = RecursiveCharacterChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk(docs)
    assert len(chunks) >= 3

    # 4. Embed & Index
    embed_model = MockEmbeddingModel(dimension=64)
    vector_store = MemoryVectorStore()
    vector_store.create_collection("police_reports", dimension=64)
    embeddings = embed_model.embed_texts([c.text for c in chunks])
    vector_store.add_chunks("police_reports", chunks, embeddings)

    # 5. Wire Pipelines
    generator = MockGeneratorLLM()
    dense_pipe = StandardDenseRAGPipeline(
        name="standard_dense_rag",
        embedding_model=embed_model,
        vector_store=vector_store,
        generator=generator,
    )
    hybrid_pipe = HybridRAGPipeline(
        name="hybrid_bm25_dense_rag",
        embedding_model=embed_model,
        vector_store=vector_store,
        generator=generator,
    )

    # 6. Evaluate
    dataset = load_ground_truth_csv(gt_csv)
    assert len(dataset) == 9

    engine = StandardEvaluationEngine(judge_type="mock")
    summary_dense = engine.evaluate(dataset=dataset, pipeline=dense_pipe, top_k=3)
    summary_hybrid = engine.evaluate(dataset=dataset, pipeline=hybrid_pipe, top_k=3)

    assert summary_dense.total_samples == 9
    assert summary_hybrid.total_samples == 9

    # Verify taxonomy coverage
    assert set(summary_dense.by_typology.keys()) == {"T1", "T2", "T3"}
    assert set(summary_dense.by_complexity.keys()) == {"C1", "C2", "C3"}

    # 7. Reporting
    md_report = generate_markdown_report(summary_dense, output_path=reports_dir / "dense_report.md")
    assert "Executive Summary" in md_report
    assert (reports_dir / "dense_report.md").exists()

    comp_report = generate_comparison_report(
        summary_dense,
        summary_hybrid,
        output_path=reports_dir / "comparison.md",
    )
    assert "Architecture Comparison" in comp_report
    assert (reports_dir / "comparison.md").exists()

    json_file = reports_dir / "summary.json"
    export_evaluation_json(summary_dense, json_file)
    assert json_file.exists()

    csv_file = reports_dir / "samples.csv"
    export_samples_csv(summary_dense, csv_file)
    assert csv_file.exists()

    q_dir = reports_dir / "questions"
    generated_q_files = generate_all_question_reports(summary_dense, output_dir=q_dir)
    assert len(generated_q_files) == 9
    assert (q_dir / "GT-T1-C1.md").exists()
    q1_text = (q_dir / "GT-T1-C1.md").read_text(encoding="utf-8")
    assert "# Benchmark Sample Report: `GT-T1-C1`" in q1_text
    assert "Executed Prompts" in q1_text
    assert "Retrieval & Chunks Breakdown" in q1_text

    # Chunks export
    chunks_dir = reports_dir / "chunks"
    generated_chunks = export_document_chunks(chunks=chunks, output_dir=chunks_dir, documents=docs)
    assert len(generated_chunks) == 3
    assert (chunks_dir / "IP_001_2023_FRAUD.txt").exists()
    assert (chunks_dir / "IP_002_2023_LAUNDERING.txt").exists()
    assert (chunks_dir / "IP_003_2023_CONSPIRACY.txt").exists()
    fraud_chunk_text = (chunks_dir / "IP_001_2023_FRAUD.txt").read_text(encoding="utf-8")
    assert "DOCUMENTO: IP_001_2023_FRAUD" in fraud_chunk_text
    assert "[CHUNK 1 /" in fraud_chunk_text
    assert "=" * 80 in fraud_chunk_text


def test_evaluate_respects_pipeline_top_k(tmp_path: Path):
    data_dir = tmp_path / "data"
    generate_sample_dataset(data_dir=data_dir)
    gt_csv = data_dir / "ground_truth.csv"
    dataset = load_ground_truth_csv(gt_csv)

    embed_model = MockEmbeddingModel(dimension=32)
    vector_store = MemoryVectorStore()
    vector_store.create_collection("police_reports", dimension=32)
    chunks = [
        Chunk(id=f"c_{i}", text=f"Chunk {i}", document_id="IP_001_2023", metadata={})
        for i in range(15)
    ]
    vector_store.add_chunks(
        "police_reports", chunks, embed_model.embed_texts([c.text for c in chunks])
    )

    pipe = StandardDenseRAGPipeline(
        embedding_model=embed_model,
        vector_store=vector_store,
        generator=MockGeneratorLLM(),
        top_k=10,
    )

    engine = StandardEvaluationEngine(judge_type="mock")
    summary = engine.evaluate(dataset=dataset[:1], pipeline=pipe)

    # When top_k is not explicitly passed to evaluate, it should default to pipe.top_k (10)
    sample = summary.samples[0]
    assert sample.response.metadata["top_k"] == 10
    assert len(sample.response.retrieved_chunks) == 10
