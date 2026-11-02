"""Unit tests for bench-rag CLI commands, focusing on skip-indexing functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bench_rag.cli import app
from bench_rag.config import PipelineConfig
from bench_rag.core.chunker import Chunk
from bench_rag.core.evaluator import (
    EvaluationSummary,
    GroundTruthItem,
    LexicalMetrics,
    LLMJudgeMetrics,
    RetrievalMetrics,
    SampleEvaluation,
)
from bench_rag.core.pipeline import RAGResponse
from bench_rag.core.vector_store import SearchResult
from bench_rag.evaluation.taxonomy import build_evaluation_summary

runner = CliRunner()


def _create_mock_summary(pipeline_name: str = "mock_pipeline") -> EvaluationSummary:
    """Create a synthetic EvaluationSummary for testing CLI reporting."""
    gt_item = GroundTruthItem(
        id="SYNTH_Q_001",
        question="What is the test parameter?",
        reference_answer="Synthetic answer.",
        typology="T1",
        complexity="C1",
        inquiry_references=["SYNTH_INQ_001"],
    )
    chunk = Chunk(
        id="c1",
        document_id="SYNTH_DOC_001",
        text="Synthetic chunk text.",
        metadata={"page_number": 1},
    )
    rag_resp = RAGResponse(
        question="What is the test parameter?",
        answer="Synthetic answer.",
        retrieved_chunks=[SearchResult(chunk=chunk, score=0.95)],
        referenced_inquiries=["SYNTH_INQ_001"],
        latency_ms=12.5,
        pipeline_name=pipeline_name,
    )
    sample = SampleEvaluation(
        item=gt_item,
        response=rag_resp,
        retrieval=RetrievalMetrics(precision_at_k=1.0, recall_at_k=1.0, hit_rate=1.0, mrr=1.0),
        lexical=LexicalMetrics(rouge_l=0.8, token_f1=0.8, bleu_4=0.8),
        llm_judge=LLMJudgeMetrics(faithfulness=1.0, answer_relevance=1.0, context_recall=1.0),
    )
    return build_evaluation_summary(
        pipeline_name=pipeline_name,
        samples=[sample],
    )


def test_evaluate_help_contains_skip_indexing():
    result = runner.invoke(app, ["evaluate", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--skip-indexing" in result.output
    assert "--skip-ingest" in result.output


def test_compare_help_contains_skip_indexing():
    result = runner.invoke(app, ["compare", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--skip-indexing" in result.output
    assert "--skip-ingest" in result.output


@patch("bench_rag.cli.load_pipeline_config")
@patch("bench_rag.cli.build_pipeline_from_config")
@patch("bench_rag.cli.load_ground_truth_csv")
@patch("bench_rag.cli.StandardEvaluationEngine")
@patch("bench_rag.cli._build_parser_from_config")
@patch("bench_rag.cli._index_chunks_with_progress")
def test_evaluate_skip_indexing_success(
    mock_index_chunks,
    mock_build_parser,
    mock_engine_cls,
    mock_load_gt,
    mock_build_pipe,
    mock_load_cfg,
    tmp_path: Path,
):
    """Test evaluate with --skip-indexing skips parsing and indexing when collection has chunks."""
    cfg = PipelineConfig(name="test_pipeline", collection_name="synth_col")
    mock_load_cfg.return_value = cfg

    mock_pipeline = MagicMock()
    mock_pipeline.vector_store.count_chunks.return_value = 15
    mock_build_pipe.return_value = mock_pipeline

    mock_load_gt.return_value = [
        GroundTruthItem(
            id="SYNTH_Q_001",
            question="Question?",
            reference_answer="Answer.",
            typology="T1",
            complexity="C1",
            inquiry_references=["SYNTH_INQ_001"],
        )
    ]

    mock_engine = MagicMock()
    mock_engine.evaluate.return_value = _create_mock_summary("test_pipeline")
    mock_engine_cls.return_value = mock_engine

    out_dir = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--skip-indexing",
            "--no-progress",
            "--output-dir",
            str(out_dir),
        ],
        env={"COLUMNS": "200"},
    )

    assert result.exit_code == 0
    assert "Skipping ingestion and indexing" in result.output
    assert "15 chunks available" in result.output
    # Ensure indexing and parsing were completely bypassed
    mock_build_parser.assert_not_called()
    mock_index_chunks.assert_not_called()
    mock_engine.evaluate.assert_called_once()
    assert "execution_metadata" in mock_engine.evaluate.call_args.kwargs
    assert mock_engine.evaluate.call_args.kwargs["execution_metadata"]["pipeline_name"] == "test_pipeline"
    # Reports should be generated, but chunks folder should NOT be exported
    assert (out_dir / "evaluation_report.html").exists()
    assert (out_dir / "run_summary.json").exists()
    assert not (out_dir / "chunks").exists()
    assert "Document chunks exported to:" not in result.output
    assert "Context Recall" in result.output


@patch("bench_rag.cli.load_pipeline_config")
@patch("bench_rag.cli.build_pipeline_from_config")
@patch("bench_rag.cli.load_ground_truth_csv")
@patch("bench_rag.cli.StandardEvaluationEngine")
@patch("bench_rag.cli._build_parser_from_config")
@patch("bench_rag.cli._index_chunks_with_progress")
def test_evaluate_skip_ingest_alias(
    mock_index_chunks,
    mock_build_parser,
    mock_engine_cls,
    mock_load_gt,
    mock_build_pipe,
    mock_load_cfg,
    tmp_path: Path,
):
    """Test that the alias --skip-ingest behaves identically to --skip-indexing."""
    cfg = PipelineConfig(name="test_pipeline", collection_name="synth_col")
    mock_load_cfg.return_value = cfg

    mock_pipeline = MagicMock()
    mock_pipeline.vector_store.count_chunks.return_value = 8
    mock_build_pipe.return_value = mock_pipeline

    mock_load_gt.return_value = [
        GroundTruthItem(
            id="SYNTH_Q_001",
            question="Question?",
            reference_answer="Answer.",
            typology="T1",
            complexity="C1",
            inquiry_references=["SYNTH_INQ_001"],
        )
    ]

    mock_engine = MagicMock()
    mock_engine.evaluate.return_value = _create_mock_summary("test_pipeline")
    mock_engine_cls.return_value = mock_engine

    out_dir = tmp_path / "reports_alias"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--skip-ingest",
            "--no-progress",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Skipping ingestion and indexing" in result.output
    assert "8 chunks available" in result.output
    mock_build_parser.assert_not_called()
    mock_index_chunks.assert_not_called()


@patch("bench_rag.cli.load_pipeline_config")
@patch("bench_rag.cli.build_pipeline_from_config")
def test_evaluate_skip_indexing_empty_collection_fails(mock_build_pipe, mock_load_cfg):
    """Test evaluate with --skip-indexing fails with code 1 if collection has 0 chunks."""
    cfg = PipelineConfig(name="test_pipeline", collection_name="empty_col")
    mock_load_cfg.return_value = cfg

    mock_pipeline = MagicMock()
    mock_pipeline.vector_store.count_chunks.return_value = 0
    mock_build_pipe.return_value = mock_pipeline

    result = runner.invoke(app, ["evaluate", "--skip-indexing", "--no-progress"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "does not exist or has 0 chunks" in result.output


def test_evaluate_no_pdfs_error_suggests_skip_indexing(tmp_path: Path):
    """Test evaluate without --skip-indexing fails when input_dir has no PDFs and suggests flag."""
    empty_dir = tmp_path / "empty_reports"
    empty_dir.mkdir()

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input-dir",
            str(empty_dir),
            "--no-progress",
        ],
    )
    assert result.exit_code == 1
    assert "No PDF reports found" in result.output
    assert "--skip-indexing" in result.output


@patch("bench_rag.cli.load_pipeline_config")
@patch("bench_rag.cli.build_pipeline_from_config")
@patch("bench_rag.cli.load_ground_truth_csv")
@patch("bench_rag.cli.StandardEvaluationEngine")
def test_compare_skip_indexing_success(
    mock_engine_cls,
    mock_load_gt,
    mock_build_pipe,
    mock_load_cfg,
    tmp_path: Path,
):
    """Test compare with --skip-indexing succeeds when both collections have chunks."""
    cfg_a = PipelineConfig(name="pipe_a", collection_name="col_a")
    cfg_b = PipelineConfig(name="pipe_b", collection_name="col_b")
    mock_load_cfg.side_effect = [cfg_a, cfg_b]

    pipe_a = MagicMock()
    pipe_a.vector_store.count_chunks.return_value = 10
    pipe_b = MagicMock()
    pipe_b.vector_store.count_chunks.return_value = 10
    mock_build_pipe.side_effect = [pipe_a, pipe_b]

    mock_load_gt.return_value = [
        GroundTruthItem(
            id="SYNTH_Q_001",
            question="Question?",
            reference_answer="Answer.",
            typology="T1",
            complexity="C1",
            inquiry_references=["SYNTH_INQ_001"],
        )
    ]

    engine_a = MagicMock()
    engine_a.evaluate.return_value = _create_mock_summary("pipe_a")
    engine_b = MagicMock()
    engine_b.evaluate.return_value = _create_mock_summary("pipe_b")
    mock_engine_cls.side_effect = [engine_a, engine_b]

    out_dir = tmp_path / "comp_out"
    result = runner.invoke(
        app,
        [
            "compare",
            "--skip-indexing",
            "--no-progress",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Skipping ingestion and indexing" in result.output
    assert "col_a" in result.output
    assert "col_b" in result.output
    # Reports generated
    assert (out_dir / "comparison_report.html").exists()
    assert not (out_dir / "chunks").exists()
    assert "Document chunks exported to:" not in result.output
    assert "Context Recall" in result.output


@patch("bench_rag.cli.load_pipeline_config")
@patch("bench_rag.cli.build_pipeline_from_config")
def test_compare_skip_indexing_empty_collection_fails(mock_build_pipe, mock_load_cfg):
    """Test compare with --skip-indexing fails if either collection is empty."""
    cfg_a = PipelineConfig(name="pipe_a", collection_name="col_a")
    cfg_b = PipelineConfig(name="pipe_b", collection_name="col_b")
    mock_load_cfg.side_effect = [cfg_a, cfg_b]

    pipe_a = MagicMock()
    pipe_a.vector_store.count_chunks.return_value = 10
    pipe_b = MagicMock()
    pipe_b.vector_store.count_chunks.return_value = 0
    mock_build_pipe.side_effect = [pipe_a, pipe_b]

    result = runner.invoke(app, ["compare", "--skip-indexing", "--no-progress"])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Collection 'col_b' for Pipeline B does not exist or has 0 chunks" in result.output
