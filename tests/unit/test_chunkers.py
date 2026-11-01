"""Unit tests for chunking algorithms."""

import pytest

from bench_rag.chunkers import RecursiveCharacterChunker, TokenChunker, get_chunker
from bench_rag.core.parser import Document


def test_recursive_chunker(sample_documents: list[Document]):
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk(sample_documents)

    assert len(chunks) > len(sample_documents)
    for c in chunks:
        assert c.document_id in {
            "IP_001_2023_FRAUD",
            "IP_002_2023_LAUNDERING",
            "IP_003_2023_CONSPIRACY",
        }
        assert len(c.text) > 0
        assert "chunk_index" in c.metadata


def test_recursive_chunker_overlap_validation():
    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        RecursiveCharacterChunker(chunk_size=100, chunk_overlap=150)


def test_recursive_chunker_empty_document():
    doc = Document(id="EMPTY_DOC", text="")
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk([doc])
    assert chunks == []


def test_token_chunker(sample_documents: list[Document]):
    chunker = TokenChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk(sample_documents)

    assert len(chunks) > 0
    for c in chunks:
        assert c.token_count <= 20
        assert len(c.text.split()) <= 20


def test_token_chunker_overlap_validation():
    with pytest.raises(ValueError, match="chunk_overlap .* must be strictly less than chunk_size"):
        TokenChunker(chunk_size=50, chunk_overlap=50)


def test_chunker_factory():
    c1 = get_chunker("recursive", chunk_size=500, chunk_overlap=50)
    assert isinstance(c1, RecursiveCharacterChunker)

    c2 = get_chunker("token", chunk_size=100, chunk_overlap=10)
    assert isinstance(c2, TokenChunker)

    with pytest.raises(ValueError, match="Unsupported chunker type"):
        get_chunker("unknown_chunker")
