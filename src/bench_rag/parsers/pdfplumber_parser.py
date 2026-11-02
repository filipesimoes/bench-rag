"""pdfplumber parser with table and layout extraction capabilities."""

from pathlib import Path
from typing import Any

import pdfplumber

from bench_rag.cleaners import HeaderFooterCleaner
from bench_rag.core.cleaner import DocumentCleaner
from bench_rag.core.parser import Document, DocumentParser


class PDFPlumberParser(DocumentParser):
    """Layout and table-aware PDF parser using pdfplumber."""

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
        """Parse a PDF report extracting textual content and formatted tables."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")

        doc_id = path.stem
        pages_content: list[str] = []
        pages_metadata: list[dict[str, Any]] = []

        with pdfplumber.open(path) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                text_parts: list[str] = []

                # Crop margins if cleaning is enabled
                target_page = page
                if self.cleaner is not None and isinstance(self.cleaner, HeaderFooterCleaner):
                    top_m = self.cleaner.header_margin
                    bot_m = self.cleaner.footer_margin
                    if (top_m > 0 or bot_m > 0) and page.height > (top_m + bot_m):
                        crop_box = (0, top_m, page.width, page.height - bot_m)
                        target_page = page.crop(crop_box)

                # Extract plain text
                raw_text = target_page.extract_text()
                if raw_text:
                    text_parts.append(raw_text.strip())

                # Extract and format tables if present
                tables = target_page.extract_tables()
                if tables:
                    for table in tables:
                        table_rows = []
                        for row in table:
                            cleaned_row = [
                                str(cell).strip() if cell is not None else "" for cell in row
                            ]
                            table_rows.append(" | ".join(cleaned_row))
                        if table_rows:
                            text_parts.append(
                                "\n[TABLE DATA]\n" + "\n".join(table_rows) + "\n[/TABLE DATA]"
                            )

                page_text = "\n\n".join(text_parts).strip()
                if page_text:
                    pages_content.append(page_text)
                    pages_metadata.append(
                        {
                            "page_number": page_idx,
                            "char_count": len(page_text),
                            "table_count": len(tables) if tables else 0,
                        }
                    )

            if self.cleaner is not None and isinstance(self.cleaner, HeaderFooterCleaner):
                cleaned_pages, meta_extras = self.cleaner.clean_document_pages_text(pages_content)
                cleaned_page_nums = meta_extras.pop("cleaned_page_numbers", None) or list(
                    range(1, len(cleaned_pages) + 1)
                )
                cleaned_pages_meta: list[dict[str, Any]] = []
                for idx_c, c_text in enumerate(cleaned_pages):
                    orig_idx = (
                        cleaned_page_nums[idx_c] - 1
                        if idx_c < len(cleaned_page_nums)
                        else idx_c
                    )
                    base_meta = (
                        dict(pages_metadata[orig_idx])
                        if 0 <= orig_idx < len(pages_metadata)
                        else {"page_number": orig_idx + 1, "table_count": 0}
                    )
                    base_meta["char_count"] = len(c_text)
                    cleaned_pages_meta.append(base_meta)

                full_text = "\n\n".join(cleaned_pages)
                metadata = {
                    "source_file": path.name,
                    "file_path": str(path.resolve()),
                    "total_pages": len(pdf.pages),
                    "pages": cleaned_pages_meta,
                    "parser": "pdfplumber",
                }
                metadata.update(meta_extras)
                return [Document(id=doc_id, text=full_text, metadata=metadata)]

            full_text = "\n\n".join(pages_content)
            metadata = {
                "source_file": path.name,
                "file_path": str(path.resolve()),
                "total_pages": len(pdf.pages),
                "pages": pages_metadata,
                "parser": "pdfplumber",
                "cleaned_headers_footers": False,
            }

            return [Document(id=doc_id, text=full_text, metadata=metadata)]
