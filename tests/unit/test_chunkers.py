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


def test_chunkers_resolve_page_number_and_strip_pages_list():
    page_1 = "Primeira pagina do relatorio sintetico com 50 caracteres."
    page_2 = "Segunda pagina do relatorio sintetico com dados adicionais."
    full_text = f"{page_1}\n\n{page_2}"
    doc = Document(
        id="IP_001_2023_FRAUD",
        text=full_text,
        metadata={
            "total_pages": 2,
            "pages": [
                {"page_number": 1, "char_count": len(page_1)},
                {"page_number": 2, "char_count": len(page_2)},
            ],
        },
    )

    rec_chunks = RecursiveCharacterChunker(chunk_size=60, chunk_overlap=0).chunk([doc])
    assert len(rec_chunks) == 2
    assert rec_chunks[0].metadata["page_number"] == 1
    assert rec_chunks[1].metadata["page_number"] == 2
    assert "pages" not in rec_chunks[0].metadata

    tok_chunks = TokenChunker(chunk_size=8, chunk_overlap=0).chunk([doc])
    assert len(tok_chunks) == 2
    assert tok_chunks[0].metadata["page_number"] == 1
    assert tok_chunks[1].metadata["page_number"] == 2
    assert "pages" not in tok_chunks[0].metadata


def test_recursive_chunker_multiline_page_offset_resolution():
    page_1 = "Paragrafo A da pagina 1.\n\nParagrafo B da pagina 1."
    page_2 = "Paragrafo C da pagina 2.\n\nParagrafo D da pagina 2."
    full_text = f"{page_1}\n\n{page_2}"
    doc = Document(
        id="IP_001_2023_FRAUD",
        text=full_text,
        metadata={
            "total_pages": 2,
            "pages": [
                {"page_number": 1, "char_count": len(page_1)},
                {"page_number": 2, "char_count": len(page_2)},
            ],
        },
    )
    chunks = RecursiveCharacterChunker(chunk_size=55, chunk_overlap=0).chunk([doc])
    assert len(chunks) == 2
    assert chunks[0].metadata["page_number"] == 1
    assert chunks[1].metadata["page_number"] == 2
    assert chunks[1].start_char >= len(page_1)

