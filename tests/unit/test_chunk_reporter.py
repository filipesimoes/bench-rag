"""Unit tests for document chunk reporting and text export."""

from pathlib import Path

from bench_rag.core.chunker import Chunk
from bench_rag.core.parser import Document
from bench_rag.reporting.chunk_reporter import (
    _sanitize_filename,
    export_document_chunks,
    format_document_chunks_txt,
)


def test_sanitize_filename():
    assert _sanitize_filename("IP_001_2023_FRAUD") == "IP_001_2023_FRAUD"
    assert _sanitize_filename("IP_001_2023_FRAUD.pdf") == "IP_001_2023_FRAUD"
    assert _sanitize_filename("invalid:characters*in?name.pdf") == "invalid_characters_in_name"
    assert _sanitize_filename("folder/subfolder/file.pdf") == "file"


def test_format_document_chunks_txt_basic():
    chunk1 = Chunk(
        id="IP_001_chk_0000",
        document_id="IP_001",
        text="Primeiro trecho do inquérito policial sobre fraude.",
        metadata={"page_number": 1, "source_file": "IP_001.pdf"},
        start_char=0,
        end_char=52,
        token_count=8,
    )
    chunk2 = Chunk(
        id="IP_001_chk_0001",
        document_id="IP_001",
        text="Segundo trecho contendo detalhes sobre transferências bancárias.",
        metadata={"page_number": 2, "source_file": "IP_001.pdf"},
        start_char=40,
        end_char=105,
        token_count=8,
    )

    doc_meta = {"source_file": "IP_001.pdf", "total_pages": 3}
    txt = format_document_chunks_txt("IP_001", [chunk1, chunk2], doc_metadata=doc_meta)

    assert "=" * 80 in txt
    assert "-" * 80 in txt
    assert "DOCUMENTO: IP_001" in txt
    assert "Arquivo de Origem: IP_001.pdf" in txt
    assert "Total de Páginas: 3" in txt
    assert "Total de Chunks Extraídos: 2" in txt
    assert "[CHUNK 1 / 2]" in txt
    assert "ID: IP_001_chk_0000" in txt
    assert "Página: 1" in txt
    assert "Tokens: 8" in txt
    assert "Offset: 0 - 52" in txt
    assert chunk1.text in txt

    assert "[CHUNK 2 / 2]" in txt
    assert "ID: IP_001_chk_0001" in txt
    assert "Página: 2" in txt
    assert chunk2.text in txt


def test_format_document_chunks_txt_empty():
    txt = format_document_chunks_txt("IP_EMPTY", [], doc_metadata={"source_file": "empty.pdf"})
    assert "DOCUMENTO: IP_EMPTY" in txt
    assert "Total de Chunks Extraídos: 0" in txt
    assert "*Nenhum chunk extraído para este documento.*" in txt


def test_format_document_chunks_txt_minimal_metadata():
    chunk = Chunk(
        id="chk_simple",
        document_id="DOC_SIMPLE",
        text="Texto simples sem metadata de pagina.",
        metadata={},
    )
    txt = format_document_chunks_txt("DOC_SIMPLE", [chunk])
    assert "DOCUMENTO: DOC_SIMPLE" in txt
    assert "[CHUNK 1 / 1]" in txt
    assert "Página:" not in txt
    assert "Tokens:" in txt
    assert "Caracteres: 37" in txt


def test_export_document_chunks_with_documents(tmp_path: Path):
    doc1 = Document(
        id="IP_001",
        text="Texto completo do doc 1",
        metadata={"source_file": "IP_001.pdf", "total_pages": 2},
    )
    doc2 = Document(
        id="IP_002",
        text="Texto completo do doc 2",
        metadata={"source_file": "IP_002.pdf", "total_pages": 1},
    )

    chunk1 = Chunk(id="c1", document_id="IP_001", text="Chunk doc 1", metadata={"page_number": 1})
    chunk2 = Chunk(id="c2", document_id="IP_002", text="Chunk doc 2", metadata={"page_number": 1})

    out_chunks_dir = tmp_path / "chunks"
    generated_files = export_document_chunks(
        chunks=[chunk1, chunk2],
        output_dir=out_chunks_dir,
        documents=[doc1, doc2],
    )

    assert len(generated_files) == 2
    assert (out_chunks_dir / "IP_001.txt").exists()
    assert (out_chunks_dir / "IP_002.txt").exists()

    content1 = (out_chunks_dir / "IP_001.txt").read_text(encoding="utf-8")
    assert "DOCUMENTO: IP_001" in content1
    assert "Chunk doc 1" in content1


def test_export_document_chunks_without_documents(tmp_path: Path):
    chunk1 = Chunk(id="c1", document_id="DOC_A", text="Texto A", metadata={"page_number": 1})
    chunk2 = Chunk(id="c2", document_id="DOC_B", text="Texto B", metadata={"page_number": 3})

    out_chunks_dir = tmp_path / "chunks"
    generated_files = export_document_chunks(
        chunks=[chunk1, chunk2],
        output_dir=out_chunks_dir,
    )

    assert len(generated_files) == 2
    assert (out_chunks_dir / "DOC_A.txt").exists()
    assert (out_chunks_dir / "DOC_B.txt").exists()
