"""Header and footer cleaner for forensic and legal PDF documents."""

import re
from collections import Counter
from typing import Any

from bench_rag.core.cleaner import DocumentCleaner
from bench_rag.core.parser import Document


class HeaderFooterCleaner(DocumentCleaner):
    """Multi-layer sanitizer for repeated headers, footers, and forensic metadata.

    Combines:
    1. Geometric margin clipping (top/bottom margins in points).
    2. Statistical multi-page repetition detection (e.g., agency letterheads).
    3. PJe electronic signature and watermark extraction / stripping.
    4. Smart first-occurrence preservation for letterheads on cover/first page.
    """

    def __init__(
        self,
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
        self.header_margin = header_margin
        self.footer_margin = footer_margin
        self.detect_repeated_headers = detect_repeated_headers
        self.repeated_header_threshold = repeated_header_threshold
        self.remove_pje_metadata = remove_pje_metadata
        self.remove_page_numbers = remove_page_numbers
        self.preserve_first_header = preserve_first_header

        # Compile regexes with MULTILINE support
        self.pje_sig_re = re.compile(
            r"Assinado eletronicamente por:\s*(.+?)(?:\s*-\s*(\d{2}/\d{2}/\d{4}(?:\s+\d{2}:\d{2}:\d{2})?))?$",
            re.IGNORECASE | re.MULTILINE,
        )
        self.pje_doc_num_re = re.compile(r"Número do documento:\s*(\d+)", re.IGNORECASE)
        self.pje_header_re = re.compile(r"^Documento id \d+.*", re.IGNORECASE)
        self.pje_footer_num_re = re.compile(r"^Num\.\s*\d+\s*-\s*Pág\.\s*\d+.*", re.IGNORECASE)
        self.pje_url_re = re.compile(r"https?://pje\S+", re.IGNORECASE)
        self.pje_system_re = re.compile(
            r"^(?:Justiça Federal.*?|PJe\s*-\s*Processo Judicial Eletrônico.*)$", re.IGNORECASE
        )

        self.page_num_re = re.compile(
            r"^(?:Página|Pág\.?)\s*\d+(?:\s*(?:de|/)\s*\d+)?\s*$", re.IGNORECASE
        )
        self.single_num_re = re.compile(r"^\d+\s*$")
        self.ipj_page_re = re.compile(r"^\d+\s*\n\s*\d{2,4}/\d{4}\s*$", re.IGNORECASE)

        self.custom_headers = [re.compile(p, re.IGNORECASE) for p in (custom_header_patterns or [])]
        self.custom_footers = [re.compile(p, re.IGNORECASE) for p in (custom_footer_patterns or [])]

    def extract_pje_signatures(self, text: str) -> list[dict[str, Any]]:
        """Extract electronic signature metadata (signatory, timestamp, document number)."""
        signatures: list[dict[str, Any]] = []
        doc_num_match = self.pje_doc_num_re.search(text)
        doc_num = doc_num_match.group(1) if doc_num_match else None

        for match in self.pje_sig_re.finditer(text):
            signatory = match.group(1).strip()
            dt = match.group(2).strip() if match.group(2) else None
            signatures.append(
                {
                    "signatory": signatory,
                    "datetime": dt,
                    "document_number": doc_num,
                }
            )
        return signatures

    def is_pje_or_pagination_noise(self, text: str) -> bool:
        """Check if a text block or line is purely PJe metadata, signature, or pagination noise."""
        clean_text = text.strip()
        if not clean_text:
            return True

        if self.remove_pje_metadata:
            if self.pje_header_re.match(clean_text):
                return True
            if self.pje_footer_num_re.match(clean_text):
                return True
            if clean_text.startswith("Assinado eletronicamente por:"):
                return True
            if self.pje_system_re.match(clean_text):
                return True
            if self.pje_url_re.match(clean_text):
                return True
            if clean_text.startswith("Número do documento:"):
                return True

        if self.remove_page_numbers:
            if self.page_num_re.match(clean_text):
                return True
            if self.single_num_re.match(clean_text):
                return True
            if self.ipj_page_re.match(clean_text):
                return True

        for pat in self.custom_headers + self.custom_footers:
            if pat.search(clean_text):
                return True

        return False

    def clean_page_blocks(
        self,
        pages_blocks: list[list[tuple]],
        page_dimensions: list[tuple[float, float]] | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Clean a list of page blocks using layout geometry, repetition detection, and regexes.

        Parameters
        ----------
        pages_blocks:
            List of pages, where each page is a list of tuples: (x0, y0, x1, y1, text, ...)
        page_dimensions:
            Optional list of (width, height) tuples for each page. Defaults to 842.0 (A4 height).

        Returns
        -------
        tuple[list[str], dict[str, Any]]:
            (cleaned_page_texts, metadata_extras)
        """
        total_pages = len(pages_blocks)
        if total_pages == 0:
            return [], {
                "cleaned_headers_footers": True,
                "pje_signatures": [],
                "repeated_headers_found": [],
                "repeated_footers_found": [],
                "characters_removed": 0,
            }

        # 1. Collect all PJe signatures before removal
        signatures: list[dict[str, Any]] = []
        seen_sig_keys = set()
        for page_idx, blocks in enumerate(pages_blocks, start=1):
            for b in blocks:
                text = b[4]
                if "Assinado eletronicamente por:" in text:
                    for sig in self.extract_pje_signatures(text):
                        key = (sig["signatory"], sig["datetime"], sig["document_number"])
                        if key not in seen_sig_keys:
                            seen_sig_keys.add(key)
                            sig["page"] = page_idx
                            signatures.append(sig)

        # 2. Detect repeated headers and footers across pages
        repeated_headers: set[str] = set()
        repeated_footers: set[str] = set()

        if self.detect_repeated_headers and total_pages > 1:
            top_counter: Counter[str] = Counter()
            bottom_counter: Counter[str] = Counter()

            for p_idx, blocks in enumerate(pages_blocks):
                h = page_dimensions[p_idx][1] if page_dimensions else 842.0
                for b in blocks:
                    y0, y1, text = b[1], b[3], b[4].strip()
                    if not text:
                        continue
                    # Ignore PJe margin watermarks during statistical detection
                    if self.is_pje_or_pagination_noise(text):
                        continue
                    if y1 <= h * 0.25:
                        top_counter[text] += 1
                    elif y0 >= h * 0.75:
                        bottom_counter[text] += 1

            min_count = max(2, int(total_pages * self.repeated_header_threshold))
            for txt, count in top_counter.items():
                if count >= min_count:
                    repeated_headers.add(txt)
            for txt, count in bottom_counter.items():
                if count >= min_count:
                    repeated_footers.add(txt)

        # 3. Clean pages
        preserved_first_headers: set[str] = set()
        cleaned_pages: list[str] = []
        total_orig_chars = 0
        total_cleaned_chars = 0

        for p_idx, blocks in enumerate(pages_blocks):
            h = page_dimensions[p_idx][1] if page_dimensions else 842.0
            page_text_blocks: list[str] = []

            for b in blocks:
                _x0, y0, _x1, y1, text = b[:5]
                total_orig_chars += len(text)
                txt = text.strip()
                if not txt:
                    continue

                # 3a. Geometric margin clipping
                if self.header_margin > 0 and y1 <= self.header_margin:
                    continue
                if self.footer_margin > 0 and y0 >= (h - self.footer_margin):
                    continue

                # 3b. Repeated header handling with first-occurrence preservation
                if txt in repeated_headers:
                    if self.preserve_first_header and txt not in preserved_first_headers:
                        preserved_first_headers.add(txt)
                    else:
                        continue

                # 3c. Repeated footer handling
                if txt in repeated_footers:
                    continue

                # 3d. PJe and pagination patterns
                if self.is_pje_or_pagination_noise(txt):
                    continue

                page_text_blocks.append(txt)

            page_content = "\n\n".join(page_text_blocks).strip()
            total_cleaned_chars += len(page_content)
            if page_content:
                cleaned_pages.append(page_content)

        chars_removed = max(0, total_orig_chars - total_cleaned_chars)
        metadata_extras = {
            "cleaned_headers_footers": True,
            "pje_signatures": signatures,
            "repeated_headers_found": sorted(repeated_headers),
            "repeated_footers_found": sorted(repeated_footers),
            "characters_removed": chars_removed,
        }

        return cleaned_pages, metadata_extras

    def clean_document_pages_text(
        self,
        pages_text: list[str],
    ) -> tuple[list[str], dict[str, Any]]:
        """Clean pages when only text strings are available without geometric bounding boxes."""
        total_pages = len(pages_text)
        if total_pages == 0:
            return [], {
                "cleaned_headers_footers": True,
                "pje_signatures": [],
                "repeated_headers_found": [],
                "repeated_footers_found": [],
                "characters_removed": 0,
            }

        # Signatures
        signatures: list[dict[str, Any]] = []
        seen_sig_keys = set()
        for page_idx, p_text in enumerate(pages_text, start=1):
            if "Assinado eletronicamente por:" in p_text:
                for sig in self.extract_pje_signatures(p_text):
                    key = (sig["signatory"], sig["datetime"], sig["document_number"])
                    if key not in seen_sig_keys:
                        seen_sig_keys.add(key)
                        sig["page"] = page_idx
                        signatures.append(sig)

        # Repetition detection on first 3 and last 3 non-empty lines
        repeated_lines: set[str] = set()
        if self.detect_repeated_headers and total_pages > 1:
            line_counter: Counter[str] = Counter()
            for p_text in pages_text:
                lines = [l.strip() for l in p_text.splitlines() if l.strip()]
                boundary_lines = lines[:3] + lines[-3:]
                for l in boundary_lines:
                    if not self.is_pje_or_pagination_noise(l):
                        line_counter[l] += 1

            min_count = max(2, int(total_pages * self.repeated_header_threshold))
            for l, count in line_counter.items():
                if count >= min_count:
                    repeated_lines.add(l)

        preserved_first_lines: set[str] = set()
        cleaned_pages: list[str] = []
        total_orig_chars = sum(len(p) for p in pages_text)
        total_cleaned_chars = 0

        for p_text in pages_text:
            lines = p_text.splitlines()
            cleaned_lines: list[str] = []

            for line in lines:
                l_strip = line.strip()
                if not l_strip:
                    continue

                if l_strip in repeated_lines:
                    if self.preserve_first_header and l_strip not in preserved_first_lines:
                        preserved_first_lines.add(l_strip)
                    else:
                        continue

                if self.is_pje_or_pagination_noise(l_strip):
                    continue

                cleaned_lines.append(line)

            page_cleaned = "\n".join(cleaned_lines).strip()
            total_cleaned_chars += len(page_cleaned)
            if page_cleaned:
                cleaned_pages.append(page_cleaned)

        metadata_extras = {
            "cleaned_headers_footers": True,
            "pje_signatures": signatures,
            "repeated_headers_found": sorted(repeated_lines),
            "repeated_footers_found": [],
            "characters_removed": max(0, total_orig_chars - total_cleaned_chars),
        }
        return cleaned_pages, metadata_extras

    def clean(self, documents: list[Document]) -> list[Document]:
        """Clean a list of Document instances."""
        cleaned_docs: list[Document] = []

        for doc in documents:
            pages_blocks = doc.metadata.get("pages_blocks")
            pages_text = doc.metadata.get("pages_text")

            if pages_blocks:
                page_dimensions = doc.metadata.get("page_dimensions")
                cleaned_pages, meta_extras = self.clean_page_blocks(pages_blocks, page_dimensions)
                cleaned_text = "\n\n".join(cleaned_pages)
            elif pages_text:
                cleaned_pages, meta_extras = self.clean_document_pages_text(pages_text)
                cleaned_text = "\n\n".join(cleaned_pages)
            else:
                # Fallback: Treat the whole text or split by double newline as pages
                lines = doc.text.split("\n\n")
                cleaned_pages, meta_extras = self.clean_document_pages_text(lines)
                cleaned_text = "\n\n".join(cleaned_pages)

            new_metadata = dict(doc.metadata)
            new_metadata.update(meta_extras)

            cleaned_docs.append(
                Document(
                    id=doc.id,
                    text=cleaned_text,
                    metadata=new_metadata,
                )
            )

        return cleaned_docs
