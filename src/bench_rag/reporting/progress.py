"""Multi-line progress reporting with Rich for files, indexing, and benchmark queries."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from bench_rag.core.evaluator import GroundTruthItem

console = Console()


def format_file_size(size_bytes: int) -> str:
    """Format bytes into a human-readable size string (B, KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def truncate_text(text: str, max_length: int = 70) -> str:
    """Cleanly truncate text with ellipsis if it exceeds max_length."""
    sanitized = " ".join(text.split())
    if len(sanitized) <= max_length:
        return sanitized
    return sanitized[: max_length - 3] + "..."


class MainOnlyColumn(ProgressColumn):
    """Wrapper that only renders the inner column for the main task, hiding it for detail lines."""

    def __init__(self, inner: ProgressColumn):
        super().__init__()
        self.inner = inner

    def get_table_column(self):
        return self.inner.get_table_column()

    def render(self, task):
        from rich.text import Text

        if task.fields.get("is_detail"):
            return Text("")
        return self.inner.render(task)


def build_progress_bar(disable: bool = False) -> Progress:
    """Build a standard Progress instance with multi-line layout support."""
    return Progress(
        MainOnlyColumn(SpinnerColumn(spinner_name="dots")),
        TextColumn("{task.description}"),
        MainOnlyColumn(BarColumn(bar_width=32)),
        MainOnlyColumn(TaskProgressColumn()),
        MainOnlyColumn(MofNCompleteColumn()),
        MainOnlyColumn(TimeElapsedColumn()),
        MainOnlyColumn(TimeRemainingColumn()),
        console=console,
        disable=disable or (not console.is_terminal),
    )


class MultiLineProgressTracker:
    """Tracks overall task progress while updating a second detail line."""

    def __init__(
        self,
        progress: Progress,
        main_task_id: TaskID,
        detail_task_id: TaskID,
        total: int,
    ):
        self.progress = progress
        self.main_task_id = main_task_id
        self.detail_task_id = detail_task_id
        self.total = total

    def update_detail(self, detail_text: str) -> None:
        """Update the lower detail line."""
        self.progress.update(
            self.detail_task_id,
            description=f"    ↳ [dim]{detail_text}[/dim]",
        )

    def advance(self, amount: int = 1) -> None:
        """Advance the main progress bar."""
        self.progress.advance(self.main_task_id, amount)

    def update_progress(self, completed: int, total: int | None = None) -> None:
        """Explicitly set completed count on the main progress task."""
        from typing import Any

        kwargs: dict[str, Any] = {"completed": completed}
        if total is not None:
            kwargs["total"] = total
            self.total = total
        self.progress.update(self.main_task_id, **kwargs)

    def evaluation_callback(
        self,
        current: int,
        total: int,
        item: GroundTruthItem,
        stage: str = "evaluating",
    ) -> None:
        """Callback compatible with EvaluationProgressCallback."""
        stage_map = {
            "rag_query": "Stage: RAG Query",
            "llm_judge": "Stage: LLM Judge",
            "completed": "Stage: Completed",
        }
        stage_label = stage_map.get(stage, f"Stage: {stage}")
        truncated_q = truncate_text(item.question, max_length=65)
        self.update_detail(f"[{stage_label}] [{item.id}] {truncated_q}")

        if stage == "completed":
            self.advance(1)


@contextmanager
def track_files(
    files: list[Path],
    description: str = "[bold cyan]Ingesting PDF reports[/bold cyan]",
    no_progress: bool = False,
) -> Generator[MultiLineProgressTracker, None, None]:
    """Context manager for tracking file processing with a multi-line progress bar."""
    progress = build_progress_bar(disable=no_progress)
    with progress:
        main_id = progress.add_task(description, total=len(files))
        detail_id = progress.add_task("    ↳ [dim]Starting...[/dim]", total=None, is_detail=True)
        tracker = MultiLineProgressTracker(
            progress=progress,
            main_task_id=main_id,
            detail_task_id=detail_id,
            total=len(files),
        )
        yield tracker


@contextmanager
def track_queries(
    total: int,
    description: str = "[bold green]Evaluating queries[/bold green]",
    no_progress: bool = False,
) -> Generator[MultiLineProgressTracker, None, None]:
    """Context manager for tracking query evaluation with a multi-line progress bar."""
    progress = build_progress_bar(disable=no_progress)
    with progress:
        main_id = progress.add_task(description, total=total)
        detail_id = progress.add_task("    ↳ [dim]Starting...[/dim]", total=None, is_detail=True)
        tracker = MultiLineProgressTracker(
            progress=progress,
            main_task_id=main_id,
            detail_task_id=detail_id,
            total=total,
        )
        yield tracker


@contextmanager
def track_embeddings(
    total: int,
    description: str = "[bold magenta]Generating embeddings[/bold magenta]",
    no_progress: bool = False,
) -> Generator[MultiLineProgressTracker, None, None]:
    """Context manager for tracking embedding generation with a multi-line progress bar."""
    progress = build_progress_bar(disable=no_progress)
    with progress:
        main_id = progress.add_task(description, total=total)
        detail_id = progress.add_task("    ↳ [dim]Starting...[/dim]", total=None, is_detail=True)
        tracker = MultiLineProgressTracker(
            progress=progress,
            main_task_id=main_id,
            detail_task_id=detail_id,
            total=total,
        )
        yield tracker


@contextmanager
def track_indexing(
    total: int,
    description: str = "[bold blue]Indexing chunks[/bold blue]",
    no_progress: bool = False,
) -> Generator[MultiLineProgressTracker, None, None]:
    """Context manager for tracking chunk embedding/indexing."""
    progress = build_progress_bar(disable=no_progress)
    with progress:
        main_id = progress.add_task(description, total=total)
        detail_id = progress.add_task("    ↳ [dim]Starting...[/dim]", total=None, is_detail=True)
        tracker = MultiLineProgressTracker(
            progress=progress,
            main_task_id=main_id,
            detail_task_id=detail_id,
            total=total,
        )
        yield tracker
