"""Core abstract protocols and dataclasses for the RAG benchmark framework."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class Document:
    """Represents an ingested document, such as a police inquiry report."""

    id: str  # Normalized document identifier, e.g. "IP_001_2023"
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol for extracting structured documents from forensic files."""

    def parse(self, file_path: Path | str) -> list[Document]:
        """Parse a document file into one or more Document instances."""
        ...
