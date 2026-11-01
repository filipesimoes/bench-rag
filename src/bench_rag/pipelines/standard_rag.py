"""Standard Dense RAG pipeline."""

import time

from bench_rag.core.embedding import EmbeddingModel
from bench_rag.core.generator import GeneratorLLM
from bench_rag.core.pipeline import RAGPipeline, RAGResponse
from bench_rag.core.reranker import Reranker
from bench_rag.core.vector_store import SearchResult, VectorStore

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert forensic intelligence analyst examining criminal police inquiry reports.\n"
    "Rules for your response:\n"
    "1. Brevity & Directness: Provide an ultra-concise answer in strictly 1 to 2 sentences. "
    "State only the requested facts directly, with zero introductory phrases, greetings, methodology descriptions, or filler.\n"
    "2. Language: Always reply in the exact same language as the question (e.g., Portuguese if asked in Portuguese, English if asked in English).\n"
    "3. Evidence Grounding: Rely strictly on the provided context evidence. If the evidence is insufficient or does not contain the answer, "
    "state directly in a single sentence that the information is not present in the provided records without speculating.\n"
    "4. Formal Citation: At the very end of your response, append the relevant document IDs in brackets: "
    "'[Fontes: <DOC_ID_1>, <DOC_ID_2>]' for Portuguese responses or '[Sources: <DOC_ID_1>, <DOC_ID_2>]' for English responses."
)


class StandardDenseRAGPipeline(RAGPipeline):
    """Standard RAG pipeline using dense vector Top-K retrieval."""

    def __init__(
        self,
        name: str = "standard_dense_rag",
        embedding_model: EmbeddingModel | None = None,
        vector_store: VectorStore | None = None,
        generator: GeneratorLLM | None = None,
        reranker: Reranker | None = None,
        rerank_fetch_k: int = 20,
        top_k: int = 5,
        collection_name: str = "police_reports",
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
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    @property
    def name(self) -> str:
        return self._name

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

        # 1. Embed query
        query_vector = self.embedding_model.embed_query(question)

        # 2. Vector search (fetch candidate pool if reranker is active)
        fetch_k = max(self.rerank_fetch_k, effective_k) if self.reranker else effective_k
        retrieved_results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=fetch_k,
        )

        # 3. Optional Rerank step
        if self.reranker:
            final_results = self.reranker.rerank(
                query=question,
                results=retrieved_results,
                top_k=effective_k,
            )
        else:
            final_results = retrieved_results[:effective_k]

        # 4. Extract unique referenced inquiry document IDs preserving retrieval rank
        referenced_inquiries: list[str] = []
        seen = set()
        for res in final_results:
            doc_id = res.chunk.document_id
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                referenced_inquiries.append(doc_id)

        # 5. Construct prompt and generate answer
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
                "reranker": self.reranker.__class__.__name__ if self.reranker else None,
                "rerank_fetch_k": fetch_k if self.reranker else effective_k,
            },
            prompt=prompt,
            system_prompt=self.system_prompt,
            pre_rerank_chunks=retrieved_results if self.reranker else [],
        )
