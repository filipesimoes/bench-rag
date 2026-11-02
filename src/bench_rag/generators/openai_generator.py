"""OpenAI-compatible LLM generator client."""

import logging
import re
import time
from typing import Any

import httpx

from bench_rag.config import settings
from bench_rag.core.generator import GenerationResult, GeneratorLLM

logger = logging.getLogger(__name__)

_UNSET = object()


class OpenAIGeneratorLLM(GeneratorLLM):
    """Client for OpenAI-compatible chat completion endpoints."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool | None | object = _UNSET,
        dialect: str = "auto",
        timeout: float | None = None,
        max_retries: int | None = None,
        retry_backoff: float | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | str | None | object = _UNSET,
    ):
        self.model = model or settings.generator_model
        resolved_url = base_url or settings.effective_generator_base_url
        self.base_url = resolved_url.rstrip("/")
        self.api_key = api_key or settings.effective_generator_api_key
        self.enable_thinking = (
            settings.generator_enable_thinking if enable_thinking is _UNSET else enable_thinking
        )
        self.dialect = dialect
        self.timeout = timeout if timeout is not None else settings.generator_timeout
        self.max_retries = max_retries if max_retries is not None else settings.http_max_retries
        self.retry_backoff = (
            retry_backoff if retry_backoff is not None else settings.http_retry_backoff
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        configured_rf = (
            getattr(settings, "generator_response_format", "json_object")
            if response_format is _UNSET
            else response_format
        )
        if isinstance(configured_rf, str):
            self.response_format = {"type": configured_rf} if configured_rf else None
        else:
            self.response_format = configured_rf

    def _resolve_dialect(self) -> str:
        """Resolve effective API dialect (auto, vllm, gemini, or openai)."""
        d = (self.dialect or "auto").lower()
        if d != "auto":
            return d
        if "generativelanguage.googleapis.com" in self.base_url:
            return "gemini"
        if "api.openai.com" in self.base_url:
            return "openai"
        return "vllm"

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | str | None | object = _UNSET,
    ) -> GenerationResult:
        effective_max_tokens = (
            max_tokens
            if max_tokens is not None
            else (self.max_tokens if self.max_tokens is not None else settings.generator_max_tokens)
        )
        effective_temperature = (
            temperature
            if temperature is not None
            else (
                self.temperature if self.temperature is not None else settings.generator_temperature
            )
        )
        effective_rf = (
            self.response_format if response_format is _UNSET else response_format
        )
        if isinstance(effective_rf, str):
            effective_rf = {"type": effective_rf} if effective_rf else None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
        }
        if effective_rf is not None:
            payload["response_format"] = effective_rf
        if self.enable_thinking is not None:
            effective_dialect = self._resolve_dialect()
            if effective_dialect in ("gemini", "openai"):
                payload["reasoning_effort"] = "high" if self.enable_thinking else "none"
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        data = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(url, json=payload, headers=headers)
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                            sleep_s = self.retry_backoff * (2**attempt)
                            logger.warning(
                                "OpenAI API call to %s returned HTTP %s (attempt %d/%d). Retrying in %.1fs...",
                                url,
                                e.response.status_code,
                                attempt + 1,
                                self.max_retries + 1,
                                sleep_s,
                            )
                            time.sleep(sleep_s)
                            continue
                        raise httpx.HTTPStatusError(
                            f"API call to {url} failed with status {e.response.status_code}: {e.response.text}",
                            request=e.request,
                            response=e.response,
                        ) from e
                    data = response.json()
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    sleep_s = self.retry_backoff * (2**attempt)
                    logger.warning(
                        "OpenAI API call to %s failed with %s: %s (attempt %d/%d). Retrying in %.1fs...",
                        url,
                        type(e).__name__,
                        str(e),
                        attempt + 1,
                        self.max_retries + 1,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                raise

        if data is None:
            raise RuntimeError(f"OpenAI API call to {url} failed to return data.")

        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            raise RuntimeError(
                f"Generation was truncated because it exceeded max_tokens ({effective_max_tokens}). "
                f"Model: {self.model}. Consider increasing GENERATOR_MAX_TOKENS or disabling thinking."
            )

        msg = choice.get("message", {})
        raw_content = msg.get("content") or ""
        answer = raw_content.strip()
        if "</think>" in answer:
            answer = re.sub(r"^.*?<\/think>\s*", "", answer, flags=re.DOTALL).strip()
        elif "<think>" in answer:
            answer = re.sub(r"<think>.*?<\/think>", "", answer, flags=re.DOTALL).strip()

        usage = data.get("usage", {})

        return GenerationResult(
            answer=answer,
            prompt=prompt,
            model_name=self.model,
            usage=usage,
            raw_response=data,
        )
