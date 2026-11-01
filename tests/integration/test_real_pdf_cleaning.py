"""Integration tests for header/footer cleaning on real police inquiry PDF reports."""

from pathlib import Path

import pytest

from bench_rag.chunkers.recursive_chunker import RecursiveCharacterChunker
from bench_rag.parsers.pymupdf_parser import PyMuPDFParser


def test_real_pdf_cleaning_ipj_2024_41():
    pdf_path = Path("data/real/teste/ipjs/ipj.2024.41.pdf")
    if not pdf_path.exists():
        pytest.skip(f"Real report {pdf_path} not found.")

    # 1. Parse raw without cleaning
    parser_raw = PyMuPDFParser(clean_headers_footers=False)
    docs_raw = parser_raw.parse(pdf_path)
    chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=200)
    chunks_raw = chunker.chunk(docs_raw)

    pje_footers_raw = sum(1 for c in chunks_raw if "Assinado eletronicamente por:" in c.text)
    pje_headers_raw = sum(1 for c in chunks_raw if "Documento id" in c.text)
    assert pje_footers_raw > 50, f"Expected many footers in raw, found {pje_footers_raw}"
    assert pje_headers_raw > 50, f"Expected many headers in raw, found {pje_headers_raw}"

    # 2. Parse with cleaning enabled
    parser_clean = PyMuPDFParser(
        clean_headers_footers=True,
        header_margin=50.0,
        footer_margin=40.0,
        detect_repeated_headers=True,
        repeated_header_threshold=0.3,
        remove_pje_metadata=True,
        remove_page_numbers=True,
        preserve_first_header=True,
    )
    docs_clean = parser_clean.parse(pdf_path)
    assert len(docs_clean) == 1
    doc = docs_clean[0]

    # Verify metadata
    assert doc.metadata["cleaned_headers_footers"] is True
    assert doc.metadata["characters_removed"] > 30000
    signatures = doc.metadata["pje_signatures"]
    assert len(signatures) >= 1
    assert all("signatory" in s and bool(s["signatory"].strip()) for s in signatures)

    # 3. Chunk cleaned document
    chunks_clean = chunker.chunk(docs_clean)

    # Verify chunk reduction and zero repeated header/footer noise
    assert len(chunks_clean) < len(chunks_raw)
    pje_footers_clean = sum(1 for c in chunks_clean if "Assinado eletronicamente por:" in c.text)
    pje_headers_clean = sum(1 for c in chunks_clean if "Documento id" in c.text)
    assert pje_footers_clean == 0
    assert pje_headers_clean == 0

    # Verify key substantive content preserved
    assert "INTRODUÇÃO E OBJETIVO" in doc.text
    assert "ESTRUTURA DA ORCRIM" in doc.text
