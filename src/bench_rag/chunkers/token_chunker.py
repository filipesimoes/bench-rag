"""Token/word-based sliding window chunker."""

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
            tokens = doc.text.split()
            if not tokens:
                continue

            chunk_idx = 0
            for i in range(0, len(tokens), self.step_size):
                window = tokens[i : i + self.chunk_size]
                if not window:
                    continue

                chunk_text = " ".join(window)
                chunk_meta = dict(doc.metadata)
                chunk_meta["chunk_index"] = chunk_idx

                chunks.append(
                    Chunk(
                        id=f"{doc.id}_token_chk_{chunk_idx:04d}",
                        document_id=doc.id,
                        text=chunk_text,
                        metadata=chunk_meta,
                        token_count=len(window),
                    )
                )
                chunk_idx += 1

                if i + self.chunk_size >= len(tokens):
                    break

        return chunks
