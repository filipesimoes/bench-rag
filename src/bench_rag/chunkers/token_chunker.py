"""Token/word-based sliding window chunker."""

import re

from bench_rag.chunkers.recursive_chunker import resolve_page_number
from bench_rag.core.chunker import Chunk, Chunker
from bench_rag.core.parser import Document


class TokenChunker(Chunker):
    """Chunks documents into sliding token/word windows."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.step_size = chunk_size - chunk_overlap

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split documents into token windows."""
        chunks: list[Chunk] = []

        for doc in documents:
            token_matches = list(re.finditer(r"\S+", doc.text))
            if not token_matches:
                continue

            pages_meta = doc.metadata.get("pages") if isinstance(doc.metadata, dict) else None
            chunk_idx = 0
            for i in range(0, len(token_matches), self.step_size):
                window_matches = token_matches[i : i + self.chunk_size]
                if not window_matches:
                    continue

                start_pos = window_matches[0].start()
                end_pos = window_matches[-1].end()
                chunk_text = " ".join(m.group(0) for m in window_matches)
                chunk_meta = dict(doc.metadata)
                chunk_meta.pop("pages", None)
                chunk_meta["chunk_index"] = chunk_idx
                if "page_number" not in chunk_meta:
                    resolved_page = resolve_page_number(start_pos, pages_meta)
                    if resolved_page is not None:
                        chunk_meta["page_number"] = resolved_page

                chunks.append(
                    Chunk(
                        id=f"{doc.id}_token_chk_{chunk_idx:04d}",
                        document_id=doc.id,
                        text=chunk_text,
                        metadata=chunk_meta,
                        start_char=start_pos,
                        end_char=end_pos,
                        token_count=len(window_matches),
                    )
                )
                chunk_idx += 1

                if i + self.chunk_size >= len(token_matches):
                    break

        return chunks
