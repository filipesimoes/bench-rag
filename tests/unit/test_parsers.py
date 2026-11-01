"""Unit tests for PDF document parsers."""

from pathlib import Path

import pytest

from bench_rag.data_gen.seed import create_pdf_report
from bench_rag.parsers import PDFPlumberParser, PyMuPDFParser, get_parser


@pytest.fixture
def temp_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "IP_999_2023_TEST.pdf"
    create_pdf_report(
        file_path=pdf_path,
        title="Test Forensic Inquiry Report",
        inquiry_id="IP-999/2023",
        sections=[
            {"title": "Section One", "content": "This is test body text for forensic analysis."},
            {
                "title": "Section Two",
                "content": [["Key", "Value"], ["Case", "999"], ["Status", "Active"]],
            },
        ],
    )
    return pdf_path


def test_pymupdf_parser(temp_pdf: Path):
    parser = PyMuPDFParser()
    docs = parser.parse(temp_pdf)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.id == "IP_999_2023_TEST"  # Normalized document identifier from filename
    assert "Test Forensic Inquiry Report" in doc.text
    assert "SECTION ONE" in doc.text
    assert doc.metadata["parser"] == "pymupdf"
    assert doc.metadata["total_pages"] >= 1


def test_pdfplumber_parser(temp_pdf: Path):
    parser = PDFPlumberParser()
    docs = parser.parse(temp_pdf)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.id == "IP_999_2023_TEST"
    assert "SECTION ONE" in doc.text
    assert doc.metadata["parser"] == "pdfplumber"


def test_parser_factory():
    p1 = get_parser("pymupdf")
    assert isinstance(p1, PyMuPDFParser)

    p2 = get_parser("pdfplumber")
    assert isinstance(p2, PDFPlumberParser)

    with pytest.raises(ValueError, match="Unsupported parser type"):
        get_parser("unknown_parser")


def test_parser_file_not_found():
    parser = PyMuPDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse("non_existent_file.pdf")
