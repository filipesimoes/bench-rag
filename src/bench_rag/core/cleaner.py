"""Core abstract protocol for document cleaners and preprocessors."""

from typing import Protocol, runtime_checkable

from bench_rag.core.parser import Document


@runtime_checkable
class DocumentCleaner(Protocol):
    """Protocol for sanitizing and preprocessing Document instances prior to chunking."""

    def clean(self, documents: list[Document]) -> list[Document]:
        """Sanitize a list of Document instances, stripping noise and repeated headers/footers."""
        ...
