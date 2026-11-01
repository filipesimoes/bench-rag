"""PyMuPDF parser for forensic police reports."""

from pathlib import Path
from typing import Any

import pymupdf

from bench_rag.cleaners import HeaderFooterCleaner
from bench_rag.core.cleaner import DocumentCleaner
from bench_rag.core.parser import Document, DocumentParser


class PyMuPDFParser(DocumentParser):
    """High-performance PDF parser using PyMuPDF."""

    def __init__(
        self,
        cleaner: DocumentCleaner | None = None,
        clean_headers_footers: bool = False,
        header_margin: float = 50.0,
        footer_margin: float = 40.0,
        detect_repeated_headers: bool = True,
        repeated_header_threshold: float = 0.3,
        remove_pje_metadata: bool = True,
        remove_page_numbers: bool = True,
        preserve_first_header: bool = True,
        custom_header_patterns: list[str] | None = None,
        custom_footer_patterns: list[str] | None = None,
    ):
        if cleaner is not None:
            self.cleaner = cleaner
        elif clean_headers_footers:
            self.cleaner = HeaderFooterCleaner(
                header_margin=header_margin,
                footer_margin=footer_margin,
                detect_repeated_headers=detect_repeated_headers,
                repeated_header_threshold=repeated_header_threshold,
                remove_pje_metadata=remove_pje_metadata,
                remove_page_numbers=remove_page_numbers,
                preserve_first_header=preserve_first_header,
                custom_header_patterns=custom_header_patterns,
                custom_footer_patterns=custom_footer_patterns,
            )
        else:
            self.cleaner = None

    def parse(self, file_path: Path | str) -> list[Document]:
        """Parse a PDF report into a normalized Document."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        # The normalized document name is derived from the PDF filename (e.g. IP_001_2023_FRAUD)
        doc_id = path.stem

        doc = pymupdf.open(path)
        pages_content: list[str] = []
        pages_metadata: list[dict[str, Any]] = []

        try:
            if self.cleaner is not None and isinstance(self.cleaner, HeaderFooterCleaner):
                pages_blocks = [page.get_text("blocks") for page in doc]
                page_dimensions = [(page.rect.width, page.rect.height) for page in doc]
                cleaned_pages, meta_extras = self.cleaner.clean_page_blocks(
                    pages_blocks, page_dimensions
                )

                for page_idx, text in enumerate(cleaned_pages, start=1):
                    if text.strip():
                        pages_content.append(text)
                        pages_metadata.append(
                            {
                                "page_number": page_idx,
                                "char_count": len(text),
                            }
                        )

                full_text = "\n\n".join(pages_content)
                metadata = {
                    "source_file": path.name,
                    "file_path": str(path.resolve()),
                    "total_pages": len(doc),
                    "pages": pages_metadata,
                    "parser": "pymupdf",
                }
                metadata.update(meta_extras)
                return [Document(id=doc_id, text=full_text, metadata=metadata)]

            for page_idx, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages_content.append(text)
                    pages_metadata.append(
                        {
                            "page_number": page_idx,
                            "char_count": len(text),
                        }
                    )

            full_text = "\n\n".join(pages_content)
            metadata = {
                "source_file": path.name,
                "file_path": str(path.resolve()),
                "total_pages": len(doc),
                "pages": pages_metadata,
                "parser": "pymupdf",
                "cleaned_headers_footers": False,
            }

            return [Document(id=doc_id, text=full_text, metadata=metadata)]
        finally:
            doc.close()
