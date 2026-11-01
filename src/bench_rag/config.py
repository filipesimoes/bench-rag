"""Configuration management using pydantic-settings and YAML."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseModel):
    """Specification for configuring a RAG pipeline instance."""

    name: str = "standard_dense_rag"
    pipeline_type: Literal["standard", "hybrid"] = "standard"
    parser_type: Literal["pymupdf", "pdfplumber"] = "pymupdf"
    # Document Preprocessing / Header & Footer Cleaning
    clean_headers_footers: bool = False
    header_margin: float = 50.0
    footer_margin: float = 40.0
    detect_repeated_headers: bool = True
    repeated_header_threshold: float = 0.3
    remove_pje_metadata: bool = True
    remove_page_numbers: bool = True
    preserve_first_header: bool = True
    custom_header_patterns: list[str] = []
    custom_footer_patterns: list[str] = []

    chunker_type: Literal["recursive", "token"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_type: Literal["openai", "mock"] = "mock"
    vector_store_type: Literal["qdrant", "memory"] = "memory"
    generator_type: Literal["openai", "mock"] = "mock"
    judge_type: Literal["openai", "mock"] = "mock"
    reranker_type: Literal["none", "mock", "http", "vllm", "tei", "cohere", "jina"] = "none"
    reranker_model: str | None = None
    reranker_base_url: str | None = None
    rerank_fetch_k: int = 20
    top_k: int = 5
    collection_name: str = "police_reports"
    distance_metric: Literal["cosine", "euclidean", "dot"] = "cosine"
    # Hybrid parameters
    rrf_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5

    # Optional endpoint and model overrides per pipeline
    generator_base_url: str | None = None
    generator_model: str | None = None
    generator_max_tokens: int | None = None
    generator_temperature: float | None = None
    generator_enable_thinking: bool | None = None
    generator_dialect: Literal["auto", "vllm", "gemini", "openai"] = "auto"
    judge_base_url: str | None = None
    judge_model: str | None = None
    judge_max_tokens: int | None = None
    judge_enable_thinking: bool | None = None
    judge_dialect: Literal["auto", "vllm", "gemini", "openai"] = "auto"
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    embedding_query_prefix: str | None = None
    embedding_document_prefix: str | None = None

    # Optional prompt customization
    system_prompt: str | None = None
    generator_response_format: str | None = None


class Settings(BaseSettings):
    """Global environment and service settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base / Fallback API Configuration
    openai_api_key: str = "mock-key"
    openai_base_url: str = "https://api.openai.com/v1"

    # Generator LLM Endpoint
    generator_base_url: str | None = None
    generator_api_key: str | None = None
    generator_model: str = "gpt-4o-mini"
    generator_max_tokens: int = 8192
    generator_temperature: float = 0.0
    generator_enable_thinking: bool | None = None
    generator_dialect: Literal["auto", "vllm", "gemini", "openai"] = "auto"
    generator_response_format: str | None = "json_object"

    # Judge LLM Endpoint
    judge_base_url: str | None = None
    judge_api_key: str | None = None
    judge_model: str = "gpt-4o-mini"
    judge_max_tokens: int = 2048
    judge_enable_thinking: bool | None = None
    judge_dialect: Literal["auto", "vllm", "gemini", "openai"] = "auto"

    # Embedding Model Endpoint
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""

    @property
    def effective_generator_base_url(self) -> str:
        return self.generator_base_url or self.openai_base_url

    @property
    def effective_generator_api_key(self) -> str:
        return self.generator_api_key or self.openai_api_key

    @property
    def effective_judge_base_url(self) -> str:
        return self.judge_base_url or self.openai_base_url

    @property
    def effective_judge_api_key(self) -> str:
        return self.judge_api_key or self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.openai_base_url

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    # Reranker Endpoint (vLLM / TEI / Cohere)
    reranker_base_url: str | None = None
    reranker_api_key: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    @property
    def effective_reranker_base_url(self) -> str:
        return self.reranker_base_url or "http://localhost:8000/v1"

    @property
    def effective_reranker_api_key(self) -> str:
        return self.reranker_api_key or self.openai_api_key

    @property
    def effective_reranker_model(self) -> str:
        return self.reranker_model

    # Qdrant Database
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "police_reports"
    qdrant_timeout: float = 60.0

    # Defaults
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    default_top_k: int = 5
    log_level: str = "INFO"


def load_pipeline_config(config_path: Path | str | None) -> PipelineConfig:
    """Load PipelineConfig from YAML file or return default configuration."""
    if config_path is None:
        return PipelineConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return PipelineConfig(**data)


# Global settings instance
settings = Settings()
