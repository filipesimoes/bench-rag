"""Hybrid RAG pipeline combining Dense Vector Search and BM25 with Reciprocal Rank Fusion (RRF)."""

import time

from rank_bm25 import BM25Okapi

from bench_rag.core.chunker import Chunk
from bench_rag.core.embedding import EmbeddingModel
from bench_rag.core.generator import GeneratorLLM
from bench_rag.core.pipeline import RAGPipeline, RAGResponse
from bench_rag.core.reranker import Reranker
from bench_rag.core.vector_store import SearchResult, VectorStore
from bench_rag.pipelines.standard_rag import DEFAULT_SYSTEM_PROMPT


class HybridRAGPipeline(RAGPipeline):
    """Hybrid RAG pipeline combining dense vector embeddings with BM25 Okapi lexical search."""

    def __init__(
        self,
        name: str = "hybrid_bm25_dense_rag",
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        generator: GeneratorLLM | None = None,
        reranker: Reranker | None = None,
        rerank_fetch_k: int = 20,
        top_k: int = 5,
        collection_name: str = "police_reports",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        system_prompt: str | None = None,
    ):
        self._name = name
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker
        self.rerank_fetch_k = rerank_fetch_k
        self.top_k = top_k
        self.collection_name = collection_name
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._bm25_index: BM25Okapi | None = None
        self._indexed_chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return self._name

    def _ensure_bm25_index(self) -> None:
        """Lazily initialize BM25 index from all indexed chunks in vector store."""
        if self._bm25_index is None:
            chunks = self.vector_store.get_all_chunks(self.collection_name)
            self._indexed_chunks = chunks
            if chunks:
                tokenized_corpus = [c.text.lower().split() for c in chunks]
                self._bm25_index = BM25Okapi(tokenized_corpus)

    def _format_context(self, results: list[SearchResult]) -> str:
        blocks = []
        for res in results:
            doc_id = res.chunk.document_id
            page = res.chunk.metadata.get("page_number", "N/A")
            blocks.append(f"[Document: {doc_id}] (Page {page}):\n{res.chunk.text}")
        return "\n\n".join(blocks)

    def query(self, question: str, top_k: int | None = None) -> RAGResponse:
        start_time = time.perf_counter()
        effective_k = top_k if top_k is not None else self.top_k
        self._ensure_bm25_index()

        # 1. Determine candidate pool size & fetch candidates
        candidate_pool_size = (
            max(self.rerank_fetch_k, effective_k) if self.reranker else effective_k
        )
        fetch_k = max(candidate_pool_size, effective_k * 2, 10)

        query_vector = self.embedding_model.embed_query(question)
        dense_results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=fetch_k,
        )

        dense_ranks: dict[str, int] = {
            res.chunk.id: rank for rank, res in enumerate(dense_results, start=1)
        }
        chunk_map: dict[str, Chunk] = {res.chunk.id: res.chunk for res in dense_results}

        # 2. BM25 Lexical retrieval
        bm25_ranks: dict[str, int] = {}
        if self._bm25_index and self._indexed_chunks:
            tokenized_query = question.lower().split()
            bm25_scores = self._bm25_index.get_scores(tokenized_query)
            # Get top indices
            top_bm25_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:fetch_k]

            for rank, idx in enumerate(top_bm25_indices, start=1):
                chunk = self._indexed_chunks[idx]
                bm25_ranks[chunk.id] = rank
                if chunk.id not in chunk_map:
                    chunk_map[chunk.id] = chunk

        # 3. Reciprocal Rank Fusion (RRF)
        all_candidate_ids = set(dense_ranks.keys()).union(set(bm25_ranks.keys()))
        rrf_scores: list[tuple[str, float]] = []

        for cid in all_candidate_ids:
            score = 0.0
            if cid in dense_ranks:
                score += self.dense_weight / (self.rrf_k + dense_ranks[cid])
            if cid in bm25_ranks:
                score += self.bm25_weight / (self.rrf_k + bm25_ranks[cid])
            rrf_scores.append((cid, score))

        rrf_scores.sort(key=lambda x: x[1], reverse=True)

        # 4. Candidate selection & optional Rerank step
        candidate_results: list[SearchResult] = []
        if self.reranker:
            candidate_results = [
                SearchResult(chunk=chunk_map[cid], score=score)
                for cid, score in rrf_scores[:candidate_pool_size]
            ]
            final_results = self.reranker.rerank(
                query=question,
                results=candidate_results,
                top_k=effective_k,
            )
        else:
            final_results = [
                SearchResult(chunk=chunk_map[cid], score=score)
                for cid, score in rrf_scores[:effective_k]
            ]

        # 5. Extract referenced document IDs
        referenced_inquiries: list[str] = []
        seen = set()
        for res in final_results:
            doc_id = res.chunk.document_id
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                referenced_inquiries.append(doc_id)

        # 6. Generate Answer
        context_str = self._format_context(final_results)
        prompt = (
            f"Context Evidence:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Instructions: Answer the question above directly in 1 to 2 sentences using strictly the context evidence provided. "
            f"Match the language of the question. Append citation brackets at the end ([Fontes: ...] or [Sources: ...]). "
            f"No introductory filler.\n\n"
            f"Answer:"
        )

        gen_result = self.generator.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            question=question,
            answer=gen_result.answer,
            retrieved_chunks=final_results,
            referenced_inquiries=referenced_inquiries,
            latency_ms=latency_ms,
            pipeline_name=self._name,
            metadata={
                "model_name": gen_result.model_name,
                "usage": gen_result.usage,
                "top_k": effective_k,
                "rrf_k": self.rrf_k,
                "reranker": self.reranker.__class__.__name__ if self.reranker else None,
                "rerank_fetch_k": candidate_pool_size if self.reranker else effective_k,
            },
            prompt=prompt,
            system_prompt=self.system_prompt,
            pre_rerank_chunks=candidate_results if self.reranker else [],
        )
