"""Parser implementations and factory."""

from bench_rag.core.cleaner import DocumentCleaner
from bench_rag.core.parser import DocumentParser
from bench_rag.parsers.pdfplumber_parser import PDFPlumberParser
from bench_rag.parsers.pymupdf_parser import PyMuPDFParser


def get_parser(
    parser_type: str = "pymupdf",
    cleaner: DocumentCleaner | None = None,
    clean_headers_footers: bool = False,
    **cleaner_kwargs,
) -> DocumentParser:
    """Factory function to instantiate a DocumentParser."""
    match parser_type.lower():
        case "pymupdf":
            return PyMuPDFParser(
                cleaner=cleaner,
                clean_headers_footers=clean_headers_footers,
                **cleaner_kwargs,
            )
        case "pdfplumber":
            return PDFPlumberParser(
                cleaner=cleaner,
                clean_headers_footers=clean_headers_footers,
                **cleaner_kwargs,
            )
        case _:
            raise ValueError(
                f"Unsupported parser type: {parser_type}. Expected 'pymupdf' or 'pdfplumber'."
            )


__all__ = ["PDFPlumberParser", "PyMuPDFParser", "get_parser"]
