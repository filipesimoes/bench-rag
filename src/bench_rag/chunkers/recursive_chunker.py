"""Recursive character chunker with configurable separators and overlap."""

from typing import Any

from bench_rag.core.chunker import Chunk, Chunker
from bench_rag.core.parser import Document


class RecursiveCharacterChunker(Chunker):
    """Splits documents recursively based on separators, target chunk size, and overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        """Split a list of Document instances into discrete Chunk objects."""
        all_chunks: list[Chunk] = []

        for doc in documents:
            doc_chunks = self._split_text(doc.text, doc.id, doc.metadata)
            all_chunks.extend(doc_chunks)

        return all_chunks

    def _split_text(
        self,
        text: str,
        document_id: str,
        doc_metadata: dict[str, Any],
    ) -> list[Chunk]:
        """Recursively splits a single text string into overlapping chunks."""
        if not text.strip():
            return []

        raw_splits = self._recursive_split(text, self.separators)
        merged_chunks = self._merge_splits(raw_splits)

        chunks: list[Chunk] = []
        current_pos = 0

        for idx, chunk_text in enumerate(merged_chunks):
            start_pos = text.find(chunk_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(chunk_text)
            current_pos = max(0, end_pos - self.chunk_overlap)

            chunk_meta = dict(doc_metadata)
            chunk_meta["chunk_index"] = idx

            chunks.append(
                Chunk(
                    id=f"{document_id}_chk_{idx:04d}",
                    document_id=document_id,
                    text=chunk_text,
                    metadata=chunk_meta,
                    start_char=start_pos,
                    end_char=end_pos,
                    token_count=max(1, len(chunk_text.split())),
                )
            )

        return chunks

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively breaks text using the first matching separator."""
        final_splits: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1 :]
                break

        splits = text.split(separator) if separator else list(text)

        for s in splits:
            if not s:
                continue
            if len(s) <= self.chunk_size:
                final_splits.append(s)
            elif new_separators:
                final_splits.extend(self._recursive_split(s, new_separators))
            else:
                # Force chunk by character slice when no smaller separators exist
                for k in range(0, len(s), self.chunk_size - self.chunk_overlap):
                    slice_text = s[k : k + self.chunk_size]
                    if slice_text:
                        final_splits.append(slice_text)

        return final_splits

    def _merge_splits(self, splits: list[str]) -> list[str]:
        """Combine small splits into chunks up to chunk_size with overlap."""
        merged: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part)
            if (
                current_len + part_len + (1 if current_chunk else 0) > self.chunk_size
                and current_chunk
            ):
                merged_text = " ".join(current_chunk).strip()
                if merged_text:
                    merged.append(merged_text)

                # Overlap: keep trailing parts up to chunk_overlap
                overlap_accum: list[str] = []
                overlap_len = 0
                for item in reversed(current_chunk):
                    if overlap_len + len(item) + 1 <= self.chunk_overlap:
                        overlap_accum.insert(0, item)
                        overlap_len += len(item) + 1
                    else:
                        break

                current_chunk = overlap_accum
                current_len = sum(len(x) + 1 for x in current_chunk)

            current_chunk.append(part)
            current_len += part_len + 1

        if current_chunk:
            final_text = " ".join(current_chunk).strip()
            if final_text:
                merged.append(final_text)

        return merged
