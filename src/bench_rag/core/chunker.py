"""Core Chunker protocol and Chunk dataclass."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from bench_rag.core.parser import Document


@dataclass
class Chunk:
    """Represents a discrete segment of a document."""

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0


@runtime_checkable
class Chunker(Protocol):
    """Protocol for partitioning documents into discrete searchable chunks."""

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of Document instances into Chunk instances."""
        ...
