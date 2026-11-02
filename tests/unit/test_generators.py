import httpx
import pytest

from bench_rag.generators import MockGeneratorLLM, OpenAIGeneratorLLM, get_generator


def test_mock_generator():
    gen = MockGeneratorLLM(model_name="mock-gen")
    prompt = (
        "Context Evidence:\n"
        "[Document: IP_001_2023_FRAUD] (Page 1):\nCarlos Mendes nominee director.\n\n"
        "Question: Who is Carlos Mendes?\n\nAnswer:"
    )
    result = gen.generate(prompt=prompt)

    assert "IP_001_2023_FRAUD" in result.answer
    assert result.model_name == "mock-gen"
    assert "prompt_tokens" in result.usage


def test_openai_generator_mocked(monkeypatch):
    gen = OpenAIGeneratorLLM(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
    )

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "Carlos Mendes is the nominee director."}}],
                "usage": {"total_tokens": 42},
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            assert url == "https://api.openai.com/v1/chat/completions"
            assert json["model"] == "gpt-4o-mini"
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    res = gen.generate("Test prompt")
    assert res.answer == "Carlos Mendes is the nominee director."
    assert res.usage["total_tokens"] == 42


def test_generator_factory():
    g1 = get_generator("mock")
    assert isinstance(g1, MockGeneratorLLM)

    g2 = get_generator("openai")
    assert isinstance(g2, OpenAIGeneratorLLM)

    g3 = get_generator("openai", base_url="http://custom-llm:8000/v1", model_name="custom-gen")
    assert isinstance(g3, OpenAIGeneratorLLM)
    assert g3.base_url == "http://custom-llm:8000/v1"
    assert g3.model == "custom-gen"
    g4 = get_generator("openai", enable_thinking=False)
    assert isinstance(g4, OpenAIGeneratorLLM)
    assert g4.enable_thinking is False

    with pytest.raises(ValueError, match="Unsupported generator type"):
        get_generator("unknown")


def test_openai_generator_enable_thinking_kwargs(monkeypatch):
    captured_payload = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "Direct response."}}],
                "usage": {"total_tokens": 10},
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            nonlocal captured_payload
            captured_payload = json
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    # 1. With enable_thinking=False on vllm endpoint
    gen_false = OpenAIGeneratorLLM(
        model="QuantTrio/Qwen3.6-27B-AWQ",
        base_url="https://vllm.example.com/v1",
        api_key="sk-test",
        enable_thinking=False,
    )
    res = gen_false.generate("Test prompt")
    assert res.answer == "Direct response."
    assert captured_payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in captured_payload

    # 2. With enable_thinking=True on Gemini endpoint (auto-detected)
    gen_gemini = OpenAIGeneratorLLM(
        model="gemini-3.8-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key="sk-test",
        enable_thinking=True,
    )
    gen_gemini.generate("Test prompt")
    assert captured_payload["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in captured_payload

    # 3. With enable_thinking=False on OpenAI endpoint (auto-detected)
    gen_openai = OpenAIGeneratorLLM(
        model="o3-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        enable_thinking=False,
    )
    gen_openai.generate("Test prompt")
    assert captured_payload["reasoning_effort"] == "none"
    assert "chat_template_kwargs" not in captured_payload

    # 4. Explicit dialect override
    gen_override = OpenAIGeneratorLLM(
        model="custom-gemini",
        base_url="http://proxy:8000/v1",
        api_key="sk-test",
        enable_thinking=True,
        dialect="gemini",
    )
    gen_override.generate("Test prompt")
    assert captured_payload["reasoning_effort"] == "high"

    # 5. With enable_thinking=None
    gen_none = OpenAIGeneratorLLM(
        model="gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        enable_thinking=None,
    )
    gen_none.generate("Test prompt")
    assert "chat_template_kwargs" not in captured_payload
    assert "reasoning_effort" not in captured_payload


def test_openai_generator_fail_fast_detailed_error(monkeypatch):
    class MockErrorResponse:
        status_code = 400
        text = '{"error": {"message": "Invalid argument: custom_field"}}'

        def raise_for_status(self):
            import httpx

            req = httpx.Request("POST", "https://api.test/v1/chat/completions")
            resp = httpx.Response(400, request=req, text=self.text)
            raise httpx.HTTPStatusError("Bad Request", request=req, response=resp)

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            return MockErrorResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    gen = OpenAIGeneratorLLM(model="test", base_url="https://api.test/v1", api_key="k")
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        gen.generate("Test")

    assert "Invalid argument: custom_field" in str(exc_info.value)


def test_openai_generator_strips_think_tags(monkeypatch):
    class MockResponse:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": self.content}}],
                "usage": {"total_tokens": 50},
            }

    current_response = ""

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            return MockResponse(current_response)

    monkeypatch.setattr("httpx.Client", MockClient)

    gen = OpenAIGeneratorLLM(model="test-model", base_url="http://test/v1", api_key="k")

    # Case 1: Standard <think>...</think> tags
    current_response = "<think>Let me ponder this question deeply...</think>The final answer is 42."
    res = gen.generate("Question")
    assert res.answer == "The final answer is 42."
    assert res.raw_response["choices"][0]["message"]["content"] == current_response

    # Case 2: Only closing </think> (as sometimes produced when prompt finishes with <think>)
    current_response = "Reflecting on evidence...\n</think>\n\nTarget answer."
    res = gen.generate("Question")
    assert res.answer == "Target answer."


def test_openai_generator_max_tokens_and_temperature_payload(monkeypatch):
    captured_payload = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"finish_reason": "stop", "message": {"content": "Answer"}}],
                "usage": {"total_tokens": 10},
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            nonlocal captured_payload
            captured_payload = json
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    # 1. Defaults to settings.generator_max_tokens (8192) and settings.generator_temperature (0.0)
    gen = OpenAIGeneratorLLM(model="test-model", base_url="http://test/v1", api_key="k")
    gen.generate("Hello")
    assert captured_payload["max_tokens"] == 8192
    assert captured_payload["temperature"] == 0.0

    # 2. Instance override
    gen_custom = OpenAIGeneratorLLM(
        model="test-model",
        base_url="http://test/v1",
        api_key="k",
        max_tokens=4096,
        temperature=0.5,
    )
    gen_custom.generate("Hello")
    assert captured_payload["max_tokens"] == 4096
    assert captured_payload["temperature"] == 0.5

    # 3. Call-level override
    gen_custom.generate("Hello", max_tokens=1024, temperature=0.8)
    assert captured_payload["max_tokens"] == 1024
    assert captured_payload["temperature"] == 0.8


def test_openai_generator_raises_on_finish_reason_length(monkeypatch):
    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "<think>Processing forensic inquiry..."},
                    }
                ],
                "usage": {"completion_tokens": 1024},
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    gen = OpenAIGeneratorLLM(
        model="QuantTrio/Qwen3.6-27B-AWQ",
        base_url="http://test/v1",
        api_key="k",
        max_tokens=1024,
    )
    with pytest.raises(
        RuntimeError, match="Generation was truncated because it exceeded max_tokens"
    ):
        gen.generate("What is the report about?")


def test_generator_factory_with_tokens_and_temperature():
    g = get_generator("openai", max_tokens=4096, temperature=0.3, timeout=90.0, max_retries=2)
    assert isinstance(g, OpenAIGeneratorLLM)
    assert g.max_tokens == 4096
    assert g.temperature == 0.3
    assert g.timeout == 90.0
    assert g.max_retries == 2


def test_openai_generator_retry_on_timeout(monkeypatch):
    calls = 0

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [{"message": {"content": "Synthetic test answer."}}],
                "usage": {"total_tokens": 10},
            }

    class MockRetryClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ReadTimeout("The read operation timed out")
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockRetryClient)

    gen = OpenAIGeneratorLLM(
        model="test-model",
        base_url="https://api.test/v1",
        api_key="k",
        max_retries=2,
        retry_backoff=0.01,
    )
    res = gen.generate("Test prompt")
    assert res.answer == "Synthetic test answer."
    assert calls == 2


def test_openai_generator_exhausts_retries(monkeypatch):
    calls = 0

    class MockFailingClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("Timeout on attempt")

    monkeypatch.setattr("httpx.Client", MockFailingClient)

    gen = OpenAIGeneratorLLM(
        model="test-model",
        base_url="https://api.test/v1",
        api_key="k",
        max_retries=2,
        retry_backoff=0.01,
    )
    with pytest.raises(httpx.ReadTimeout):
        gen.generate("Test prompt")

    assert calls == 3  # 1 initial + 2 retries
