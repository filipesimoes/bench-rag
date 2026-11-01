"""Unit tests for StandardDenseRAGPipeline and HybridRAGPipeline with optional reranking."""

from bench_rag.config import PipelineConfig
from bench_rag.core.chunker import Chunk
from bench_rag.embeddings.mock_embedding import MockEmbeddingModel
from bench_rag.generators.mock_generator import MockGeneratorLLM
from bench_rag.pipelines import build_pipeline_from_config
from bench_rag.pipelines.hybrid_rag import HybridRAGPipeline
from bench_rag.pipelines.standard_rag import StandardDenseRAGPipeline
from bench_rag.rerankers.mock_reranker import MockReranker
from bench_rag.vector_stores.memory_store import MemoryVectorStore


def _setup_store_with_chunks() -> tuple[MemoryVectorStore, MockEmbeddingModel]:
    store = MemoryVectorStore()
    embedding_model = MockEmbeddingModel(dimension=32)
    store.create_collection("police_reports", dimension=32)

    chunks = [
        Chunk(
            id=f"chunk_{i}",
            text=f"Evidência documental {i}: "
            + (
                "Carlos Eduardo é o beneficiário final da conta."
                if i == 7
                else f"Registro geral de inquérito {i}."
            ),
            document_id=f"IP_00{i}_2023",
            metadata={"page_number": i},
        )
        for i in range(1, 11)
    ]
    embeddings = embedding_model.embed_texts([c.text for c in chunks])
    store.add_chunks("police_reports", chunks, embeddings)
    return store, embedding_model


def test_standard_dense_rag_with_rerank():
    store, embedding = _setup_store_with_chunks()
    generator = MockGeneratorLLM()
    reranker = MockReranker()

    pipeline = StandardDenseRAGPipeline(
        embedding_model=embedding,
        vector_store=store,
        generator=generator,
        reranker=reranker,
        rerank_fetch_k=10,
        collection_name="police_reports",
    )

    query = "Quem é o beneficiário final Carlos Eduardo?"
    response = pipeline.query(question=query, top_k=3)

    assert len(response.retrieved_chunks) == 3
    assert response.metadata["reranker"] == "MockReranker"
    assert response.metadata["rerank_fetch_k"] == 10
    # Chunk 7 contains 'Carlos Eduardo é o beneficiário final' and must be ranked 1st
    top_chunk = response.retrieved_chunks[0].chunk
    assert top_chunk.id == "chunk_7"
    assert top_chunk.metadata["initial_retrieval_rank"] > 0
    assert "IP_007_2023" in response.referenced_inquiries


def test_standard_dense_rag_without_rerank_backward_compatible():
    store, embedding = _setup_store_with_chunks()
    generator = MockGeneratorLLM()

    pipeline = StandardDenseRAGPipeline(
        embedding_model=embedding,
        vector_store=store,
        generator=generator,
        reranker=None,
        collection_name="police_reports",
    )

    response = pipeline.query(question="Alguma pergunta", top_k=3)
    assert len(response.retrieved_chunks) == 3
    assert response.metadata["reranker"] is None
    assert response.metadata["rerank_fetch_k"] == 3


def test_hybrid_rag_with_rerank():
    store, embedding = _setup_store_with_chunks()
    generator = MockGeneratorLLM()
    reranker = MockReranker()

    pipeline = HybridRAGPipeline(
        embedding_model=embedding,
        vector_store=store,
        generator=generator,
        reranker=reranker,
        rerank_fetch_k=10,
        collection_name="police_reports",
    )

    query = "Carlos Eduardo beneficiário final"
    response = pipeline.query(question=query, top_k=3)

    assert len(response.retrieved_chunks) == 3
    assert response.metadata["reranker"] == "MockReranker"
    assert response.metadata["rerank_fetch_k"] == 10
    assert response.retrieved_chunks[0].chunk.id == "chunk_7"


def test_build_pipeline_from_config_with_reranker():
    cfg = PipelineConfig(
        name="test_pipeline_rerank",
        pipeline_type="hybrid",
        embedding_type="mock",
        vector_store_type="memory",
        generator_type="mock",
        reranker_type="mock",
        rerank_fetch_k=15,
        top_k=4,
    )
    pipeline = build_pipeline_from_config(cfg)
    assert isinstance(pipeline, HybridRAGPipeline)
    assert pipeline.reranker is not None
    assert isinstance(pipeline.reranker, MockReranker)
    assert pipeline.rerank_fetch_k == 15
    assert pipeline.top_k == 4


def test_build_pipeline_from_config_standard_dense_top_k():
    cfg = PipelineConfig(
        name="test_standard_dense",
        pipeline_type="standard",
        embedding_type="mock",
        vector_store_type="memory",
        generator_type="mock",
        top_k=8,
    )
    pipeline = build_pipeline_from_config(cfg)
    assert isinstance(pipeline, StandardDenseRAGPipeline)
    assert pipeline.top_k == 8


def test_pipeline_query_uses_configured_top_k_by_default():
    store, embedding = _setup_store_with_chunks()
    generator = MockGeneratorLLM()

    pipeline = StandardDenseRAGPipeline(
        embedding_model=embedding,
        vector_store=store,
        generator=generator,
        collection_name="police_reports",
        top_k=7,
    )

    # Calling query without passing top_k should use pipeline.top_k (7)
    res_default = pipeline.query("alguma pergunta")
    assert len(res_default.retrieved_chunks) == 7
    assert res_default.metadata["top_k"] == 7

    # Explicit override should still work
    res_override = pipeline.query("alguma pergunta", top_k=2)
    assert len(res_override.retrieved_chunks) == 2
    assert res_override.metadata["top_k"] == 2


def test_concise_prompt_construction_and_system_prompt():
    store, embedding = _setup_store_with_chunks()
    captured_prompts: dict[str, str] = {}

    class CapturingGenerator:
        model_name = "capturing-gen"

        def generate(self, prompt: str, system_prompt: str | None = None, **kwargs):
            from bench_rag.core.generator import GenerationResult

            captured_prompts["prompt"] = prompt
            captured_prompts["system_prompt"] = system_prompt
            return GenerationResult(
                answer="Carlos Eduardo é o beneficiário. [Fontes: IP_007_2023]",
                prompt=prompt,
                model_name=self.model_name,
            )

    gen = CapturingGenerator()
    pipe = StandardDenseRAGPipeline(
        embedding_model=embedding,
        vector_store=store,
        generator=gen,
        collection_name="police_reports",
        top_k=2,
    )

    # 1. Check default system_prompt rules
    assert "Rules for your response:" in pipe.system_prompt
    assert "1 to 2 sentences" in pipe.system_prompt
    assert "[Fontes: <DOC_ID_1>, <DOC_ID_2>]" in pipe.system_prompt
    assert "[Sources: <DOC_ID_1>, <DOC_ID_2>]" in pipe.system_prompt

    # 2. Query and inspect constructed user prompt
    res = pipe.query("Quem é o beneficiário?")
    assert "Investigative Analysis and Findings:" not in captured_prompts["prompt"]
    assert (
        "Instructions: Answer the question above directly in 1 to 2 sentences"
        in captured_prompts["prompt"]
    )
    assert "Append citation brackets at the end" in captured_prompts["prompt"]
    assert captured_prompts["prompt"].endswith("Answer:")
    assert captured_prompts["system_prompt"] == pipe.system_prompt
    assert "[Fontes: IP_007_2023]" in res.answer


def test_build_pipeline_from_config_system_prompt_forwarding():
    cfg_custom = PipelineConfig(
        pipeline_type="standard",
        embedding_type="mock",
        vector_store_type="memory",
        generator_type="mock",
        system_prompt="Custom analyst prompt for test.",
    )
    pipe_custom = build_pipeline_from_config(cfg_custom)
    assert pipe_custom.system_prompt == "Custom analyst prompt for test."

    cfg_hybrid = PipelineConfig(
        pipeline_type="hybrid",
        embedding_type="mock",
        vector_store_type="memory",
        generator_type="mock",
        system_prompt="Custom hybrid prompt.",
    )
    pipe_hybrid = build_pipeline_from_config(cfg_hybrid)
    assert pipe_hybrid.system_prompt == "Custom hybrid prompt."
