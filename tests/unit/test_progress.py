"""Unit tests for progress tracking and callback integration."""

from pathlib import Path
from unittest.mock import MagicMock

from bench_rag.core.chunker import Chunk
from bench_rag.core.evaluator import GroundTruthItem
from bench_rag.core.pipeline import RAGPipeline, RAGResponse
from bench_rag.core.vector_store import SearchResult
from bench_rag.evaluation.engine import StandardEvaluationEngine
from bench_rag.evaluation.metrics.llm_judge import get_judge
from bench_rag.reporting.progress import (
    MultiLineProgressTracker,
    format_file_size,
    track_embeddings,
    track_files,
    track_indexing,
    track_queries,
    truncate_text,
)


def test_format_file_size():
    assert format_file_size(500) == "500 B"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(10 * 1024 * 1024) == "10.0 MB"
    assert format_file_size(2 * 1024 * 1024 * 1024) == "2.00 GB"


def test_truncate_text():
    short = "Who is Carlos Mendes?"
    assert truncate_text(short, max_length=50) == short

    multiline = "Who is   Carlos\n  Mendes?"
    assert truncate_text(multiline, max_length=50) == "Who is Carlos Mendes?"

    long_text = "A" * 100
    truncated = truncate_text(long_text, max_length=20)
    assert len(truncated) == 20
    assert truncated.endswith("...")
    assert truncated == ("A" * 17) + "..."


def test_evaluation_engine_calls_progress_callback():
    pipeline = MagicMock(spec=RAGPipeline)
    pipeline.name = "test_pipeline"
    mock_chunk = Chunk(id="c1", document_id="IP_001", text="evidence", metadata={})
    pipeline.query.return_value = RAGResponse(
        question="What happened?",
        answer="An inquiry occurred.",
        retrieved_chunks=[SearchResult(chunk=mock_chunk, score=0.9)],
        referenced_inquiries=["IP_001"],
        latency_ms=10.0,
        pipeline_name="test_pipeline",
    )

    dataset = [
        GroundTruthItem(
            id="GT-01",
            typology="T1",
            complexity="C1",
            question="What happened?",
            reference_answer="An inquiry occurred.",
            inquiry_references=["IP_001"],
        ),
        GroundTruthItem(
            id="GT-02",
            typology="T2",
            complexity="C2",
            question="Who did it?",
            reference_answer="Subject A did it.",
            inquiry_references=["IP_001"],
        ),
    ]

    engine = StandardEvaluationEngine(judge=get_judge("mock"))

    stages_observed: list[tuple[int, int, str, str]] = []

    def mock_callback(current: int, total: int, item: GroundTruthItem, stage: str):
        stages_observed.append((current, total, item.id, stage))

    summary = engine.evaluate(dataset=dataset, pipeline=pipeline, progress_callback=mock_callback)

    assert summary.total_samples == 2
    assert stages_observed == [
        (1, 2, "GT-01", "rag_query"),
        (1, 2, "GT-01", "llm_judge"),
        (1, 2, "GT-01", "completed"),
        (2, 2, "GT-02", "rag_query"),
        (2, 2, "GT-02", "llm_judge"),
        (2, 2, "GT-02", "completed"),
    ]


def test_multiline_tracker_callbacks():
    progress_mock = MagicMock()
    tracker = MultiLineProgressTracker(
        progress=progress_mock,
        main_task_id=1,  # type: ignore
        detail_task_id=2,  # type: ignore
        total=5,
    )

    item = GroundTruthItem(
        id="GT-01",
        typology="T1",
        complexity="C1",
        question="What smurfing technique was utilized by Orion Logistics?",
        reference_answer="Smurfing below $10,000.",
        inquiry_references=["IP_002"],
    )

    tracker.evaluation_callback(1, 5, item, stage="rag_query")
    assert progress_mock.update.called
    assert not progress_mock.advance.called

    tracker.evaluation_callback(1, 5, item, stage="completed")
    progress_mock.advance.assert_called_with(1, 1)


def test_track_context_managers_with_no_progress():
    files = [Path("test1.pdf"), Path("test2.pdf")]
    with track_files(files, no_progress=True) as tracker:
        tracker.update_detail("testing")
        tracker.advance(1)

    with track_queries(total=2, no_progress=True) as tracker:
        tracker.update_detail("testing query")
        tracker.advance(1)

    with track_indexing(total=10, no_progress=True) as tracker:
        tracker.update_detail("testing indexing")
        tracker.advance(10)

    with track_embeddings(total=10, no_progress=True) as tracker:
        tracker.update_detail("testing embedding")
        tracker.advance(5)
        tracker.update_progress(10)


def test_multiline_tracker_update_progress():
    progress_mock = MagicMock()
    tracker = MultiLineProgressTracker(
        progress=progress_mock,
        main_task_id=1,  # type: ignore
        detail_task_id=2,  # type: ignore
        total=50,
    )
    tracker.update_progress(completed=20)
    progress_mock.update.assert_called_with(1, completed=20)

    tracker.update_progress(completed=40, total=60)
    progress_mock.update.assert_called_with(1, completed=40, total=60)
    assert tracker.total == 60
