"""Generator implementations and factory."""

from typing import Any

from bench_rag.config import settings
from bench_rag.core.generator import GeneratorLLM
from bench_rag.generators.mock_generator import MockGeneratorLLM
from bench_rag.generators.openai_generator import OpenAIGeneratorLLM


def get_generator(
    generator_type: str = "mock",
    model_name: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    enable_thinking: bool | None = None,
    dialect: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
    response_format: dict[str, Any] | str | None = None,
) -> GeneratorLLM:
    """Factory function to instantiate a GeneratorLLM."""
    match generator_type.lower():
        case "mock":
            return MockGeneratorLLM(model_name=model_name or "mock-generator")
        case "openai":
            kwargs: dict[str, Any] = {}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            if response_format is not None:
                kwargs["response_format"] = response_format
            kwargs["dialect"] = dialect or settings.generator_dialect
            return OpenAIGeneratorLLM(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                **kwargs,
            )
        case _:
            raise ValueError(
                f"Unsupported generator type: {generator_type}. Expected 'mock' or 'openai'."
            )


__all__ = ["MockGeneratorLLM", "OpenAIGeneratorLLM", "get_generator"]
