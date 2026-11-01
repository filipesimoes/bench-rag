"""Unit tests verifying separation of endpoints for Generator LLM, Judge LLM, and Embedding."""

from bench_rag.config import PipelineConfig, Settings
from bench_rag.embeddings.openai_embedding import OpenAIEmbeddingModel
from bench_rag.evaluation.engine import StandardEvaluationEngine
from bench_rag.evaluation.metrics.llm_judge import get_judge
from bench_rag.generators.openai_generator import OpenAIGeneratorLLM
from bench_rag.pipelines import build_pipeline_from_config


def test_settings_endpoints_separation(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fallback:8000/v1")
    monkeypatch.setenv("GENERATOR_BASE_URL", "http://generator-llm:8001/v1")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://judge-llm:8002/v1")
    monkeypatch.setenv("EMBEDDING_BASE_URL", "http://embedding-model:8003/v1")

    s = Settings()
    assert s.effective_generator_base_url == "http://generator-llm:8001/v1"
    assert s.effective_judge_base_url == "http://judge-llm:8002/v1"
    assert s.effective_embedding_base_url == "http://embedding-model:8003/v1"


def test_settings_endpoints_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fallback:8000/v1")
    monkeypatch.delenv("GENERATOR_BASE_URL", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.delenv("EMBEDDING_BASE_URL", raising=False)

    s = Settings(generator_base_url=None, judge_base_url=None, embedding_base_url=None)
    assert s.effective_generator_base_url == "http://fallback:8000/v1"
    assert s.effective_judge_base_url == "http://fallback:8000/v1"
    assert s.effective_embedding_base_url == "http://fallback:8000/v1"


def test_components_use_separated_endpoints(monkeypatch):
    monkeypatch.setattr("bench_rag.config.settings.generator_base_url", "http://gen-srv:9001/v1")
    monkeypatch.setattr("bench_rag.config.settings.judge_base_url", "http://judge-srv:9002/v1")
    monkeypatch.setattr("bench_rag.config.settings.embedding_base_url", "http://emb-srv:9003/v1")

    gen = OpenAIGeneratorLLM()
    assert gen.base_url == "http://gen-srv:9001/v1"

    emb = OpenAIEmbeddingModel()
    assert emb.base_url == "http://emb-srv:9003/v1"

    judge = get_judge("openai")
    assert isinstance(judge.generator, OpenAIGeneratorLLM)
    assert judge.generator.base_url == "http://judge-srv:9002/v1"


def test_pipeline_config_endpoint_overrides():
    cfg = PipelineConfig(
        generator_type="openai",
        generator_base_url="http://override-gen:10001/v1",
        embedding_type="openai",
        embedding_base_url="http://override-emb:10002/v1",
        judge_type="openai",
        judge_base_url="http://override-judge:10003/v1",
        judge_model="gpt-4o-judge",
    )

    pipeline = build_pipeline_from_config(cfg)
    assert isinstance(pipeline.generator, OpenAIGeneratorLLM)
    assert pipeline.generator.base_url == "http://override-gen:10001/v1"

    assert isinstance(pipeline.embedding_model, OpenAIEmbeddingModel)
    assert pipeline.embedding_model.base_url == "http://override-emb:10002/v1"

    engine = StandardEvaluationEngine(
        judge_type=cfg.judge_type,
        judge_model=cfg.judge_model,
        judge_base_url=cfg.judge_base_url,
    )
    assert isinstance(engine.judge.generator, OpenAIGeneratorLLM)
    assert engine.judge.generator.base_url == "http://override-judge:10003/v1"
    assert engine.judge.generator.model == "gpt-4o-judge"


def test_settings_embedding_prefixes(monkeypatch):
    """Test that Settings loads embedding prefixes from env vars with empty default."""
    monkeypatch.delenv("EMBEDDING_QUERY_PREFIX", raising=False)
    monkeypatch.delenv("EMBEDDING_DOCUMENT_PREFIX", raising=False)
    s_default = Settings(_env_file=None)
    assert s_default.embedding_query_prefix == ""
    assert s_default.embedding_document_prefix == ""

    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "search_query: ")
    monkeypatch.setenv("EMBEDDING_DOCUMENT_PREFIX", "search_document: ")
    s_custom = Settings(_env_file=None)
    assert s_custom.embedding_query_prefix == "search_query: "
    assert s_custom.embedding_document_prefix == "search_document: "


def test_pipeline_config_prefix_overrides():
    """Test that PipelineConfig allows overriding prefixes per pipeline."""
    cfg = PipelineConfig(
        embedding_type="openai",
        embedding_query_prefix="task_q: ",
        embedding_document_prefix="task_d: ",
    )
    assert cfg.embedding_query_prefix == "task_q: "
    assert cfg.embedding_document_prefix == "task_d: "

    pipeline = build_pipeline_from_config(cfg)
    assert isinstance(pipeline.embedding_model, OpenAIEmbeddingModel)
    assert pipeline.embedding_model.query_prefix == "task_q: "
    assert pipeline.embedding_model.document_prefix == "task_d: "


def test_pipeline_config_thinking_overrides():
    """Test that PipelineConfig allows overriding enable_thinking and dialect per pipeline."""
    cfg = PipelineConfig(
        generator_type="openai",
        generator_enable_thinking=False,
        generator_dialect="vllm",
        judge_type="openai",
        judge_enable_thinking=True,
        judge_dialect="gemini",
    )
    assert cfg.generator_enable_thinking is False
    assert cfg.judge_enable_thinking is True
    assert cfg.generator_dialect == "vllm"
    assert cfg.judge_dialect == "gemini"

    pipeline = build_pipeline_from_config(cfg)
    assert isinstance(pipeline.generator, OpenAIGeneratorLLM)
    assert pipeline.generator.enable_thinking is False
    assert pipeline.generator.dialect == "vllm"

    engine = StandardEvaluationEngine(
        judge_type=cfg.judge_type,
        judge_enable_thinking=cfg.judge_enable_thinking,
        judge_dialect=cfg.judge_dialect,
    )
    assert isinstance(engine.judge.generator, OpenAIGeneratorLLM)
    assert engine.judge.generator.enable_thinking is True
    assert engine.judge.generator.dialect == "gemini"
