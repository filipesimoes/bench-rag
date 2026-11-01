"""Export extracted document chunks to plain-text files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from bench_rag.core.chunker import Chunk
from bench_rag.core.parser import Document

_SEPARATOR_LINE = "=" * 80
_INNER_SEPARATOR_LINE = "-" * 80


def _sanitize_filename(name: str) -> str:
    """Sanitize document ID for safe filesystem filename generation."""
    base = Path(name).stem if name.lower().endswith(".pdf") else name
    safe = re.sub(r"[^\w\-.]", "_", base)
    return safe or "document"


def format_document_chunks_txt(
    doc_id: str,
    chunks: list[Chunk],
    doc_metadata: dict[str, Any] | None = None,
) -> str:
    """Format all chunks of a document into a structured plain-text representation.

    Args:
        doc_id: Normalized document identifier.
        chunks: List of Chunk objects belonging to this document.
        doc_metadata: Optional document-level metadata (e.g. source file, pages).

    Returns:
        Structured string representation with visual banner separators.
    """
    meta = doc_metadata or {}
    source_file = meta.get("source_file") or f"{doc_id}.pdf"
    total_pages = meta.get("total_pages")

    header_lines = [
        _SEPARATOR_LINE,
        f"DOCUMENTO: {doc_id}",
        f"Arquivo de Origem: {source_file}",
    ]
    if total_pages is not None:
        header_lines.append(f"Total de Páginas: {total_pages}")
    header_lines.append(f"Total de Chunks Extraídos: {len(chunks)}")
    header_lines.append(_SEPARATOR_LINE)
    header_lines.append("")

    if not chunks:
        header_lines.append("*Nenhum chunk extraído para este documento.*")
        header_lines.append("")
        return "\n".join(header_lines)

    body_lines: list[str] = []
    total_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, start=1):
        # Chunk header
        body_lines.append(_SEPARATOR_LINE)
        body_lines.append(f"[CHUNK {idx} / {total_chunks}]")
        body_lines.append(f"ID: {chunk.id}")

        meta_parts: list[str] = []
        page = chunk.metadata.get("page_number") or chunk.metadata.get("page")
        if page is not None:
            meta_parts.append(f"Página: {page}")

        tokens = chunk.token_count if chunk.token_count > 0 else max(1, len(chunk.text.split()))
        meta_parts.append(f"Tokens: {tokens}")

        char_len = len(chunk.text)
        meta_parts.append(f"Caracteres: {char_len}")

        if chunk.end_char > 0:
            meta_parts.append(f"Offset: {chunk.start_char} - {chunk.end_char}")

        body_lines.append(" | ".join(meta_parts))
        body_lines.append(_INNER_SEPARATOR_LINE)
        body_lines.append(chunk.text)
        body_lines.append(_SEPARATOR_LINE)
        body_lines.append("")

    return "\n".join(header_lines + body_lines)


def export_document_chunks(
    chunks: list[Chunk],
    output_dir: Path | str,
    documents: list[Document] | None = None,
) -> list[Path]:
    """Export document chunks to individual plain-text files inside output_dir.

    Args:
        chunks: List of Chunk objects extracted from documents.
        output_dir: Target directory path (e.g. reports/run_default/chunks).
        documents: Optional list of Document objects to preserve document order and metadata.

    Returns:
        List of Paths of created txt files.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        chunks_by_doc[c.document_id].append(c)

    generated_paths: list[Path] = []
    processed_doc_ids: set[str] = set()

    if documents is not None:
        for doc in documents:
            processed_doc_ids.add(doc.id)
            doc_chunks = chunks_by_doc.get(doc.id, [])
            text_content = format_document_chunks_txt(
                doc_id=doc.id,
                chunks=doc_chunks,
                doc_metadata=doc.metadata,
            )
            file_path = target_dir / f"{_sanitize_filename(doc.id)}.txt"
            file_path.write_text(text_content, encoding="utf-8")
            generated_paths.append(file_path)

    # Export any chunks belonging to doc_ids not present in documents
    for doc_id, doc_chunks in chunks_by_doc.items():
        if doc_id not in processed_doc_ids:
            # Attempt to retrieve doc metadata from first chunk if available
            doc_meta = dict(doc_chunks[0].metadata) if doc_chunks else {}
            text_content = format_document_chunks_txt(
                doc_id=doc_id,
                chunks=doc_chunks,
                doc_metadata=doc_meta,
            )
            file_path = target_dir / f"{_sanitize_filename(doc_id)}.txt"
            file_path.write_text(text_content, encoding="utf-8")
            generated_paths.append(file_path)

    return generated_paths
