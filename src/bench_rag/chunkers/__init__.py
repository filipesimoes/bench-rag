"""Chunker implementations and factory."""

from bench_rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from bench_rag.chunkers.token_chunker import TokenChunker
from bench_rag.core.chunker import Chunker


def get_chunker(
    chunker_type: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chunker:
    """Factory function to instantiate a Chunker."""
    match chunker_type.lower():
        case "recursive":
            return RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        case "token":
            return TokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        case _:
            raise ValueError(
                f"Unsupported chunker type: {chunker_type}. Expected 'recursive' or 'token'."
            )


__all__ = ["RecursiveCharacterChunker", "TokenChunker", "get_chunker"]
