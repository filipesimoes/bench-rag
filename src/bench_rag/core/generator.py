"""Core GeneratorLLM protocol and GenerationResult dataclass."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class GenerationResult:
    """Represents the output of an LLM generation."""

    answer: str
    prompt: str
    model_name: str
    usage: dict[str, int] = field(default_factory=dict)
    raw_response: Any | None = None


@runtime_checkable
class GeneratorLLM(Protocol):
    """Protocol for LLM generation targeting OpenAI-compatible endpoints."""

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate text given prompt and system instructions."""
        ...
