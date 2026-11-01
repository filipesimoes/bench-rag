"""Unit tests for embedding models."""

import numpy as np
import pytest

from bench_rag.embeddings import MockEmbeddingModel, OpenAIEmbeddingModel, get_embedding_model


def test_mock_embedding_model():
    model = MockEmbeddingModel(dimension=128)
    assert model.dimension == 128

    texts = ["Police report on Carlos Mendes.", "Investigation into Orion Logistics."]
    embeddings = model.embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 128
    assert len(embeddings[1]) == 128

    # Check normalization: L2 norm should be approximately 1.0
    norm_0 = np.linalg.norm(np.array(embeddings[0]))
    assert pytest.approx(norm_0, rel=1e-3) == 1.0

    # Determinism: same input produces same vector
    emb_repeat = model.embed_texts(["Police report on Carlos Mendes."])
    assert embeddings[0] == emb_repeat[0]

    # Query embedding
    q_emb = model.embed_query("Carlos Mendes")
    assert len(q_emb) == 128


def test_openai_embedding_mocked(monkeypatch):
    model = OpenAIEmbeddingModel(
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        dimension=128,
        batch_size=2,
    )

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "data": [
                    {"index": 0, "embedding": [0.1] * 128},
                    {"index": 1, "embedding": [0.2] * 128},
                ]
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            assert url == "https://api.openai.com/v1/embeddings"
            assert json["model"] == "text-embedding-3-small"
            assert len(json["input"]) <= 2
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    res = model.embed_texts(["Text 1", "Text 2"])
    assert len(res) == 2
    assert len(res[0]) == 128


def test_embedding_factory():
    m1 = get_embedding_model("mock", dimension=256)
    assert isinstance(m1, MockEmbeddingModel)
    assert m1.dimension == 256

    m2 = get_embedding_model("openai")
    assert isinstance(m2, OpenAIEmbeddingModel)

    m3 = get_embedding_model(
        "openai",
        base_url="http://custom-emb:8001/v1",
        model_name="custom-emb-model",
        dimension=768,
    )
    assert isinstance(m3, OpenAIEmbeddingModel)
    assert m3.base_url == "http://custom-emb:8001/v1"
    assert m3.model == "custom-emb-model"
    assert m3.dimension == 768

    with pytest.raises(ValueError, match="Unsupported embedding type"):
        get_embedding_model("invalid")


def test_openai_embedding_prefixes_concatenation(monkeypatch):
    """Test that query_prefix and document_prefix are properly prepended in OpenAIEmbeddingModel."""
    model = OpenAIEmbeddingModel(
        model="nomic-ai/nomic-embed-text-v1.5",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        dimension=128,
        batch_size=2,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )

    captured_inputs = []

    class MockResponse:
        def __init__(self, count):
            self.count = count

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": i, "embedding": [0.1] * 128} for i in range(self.count)]}

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            captured_inputs.append(json["input"])
            return MockResponse(len(json["input"]))

    monkeypatch.setattr("httpx.Client", MockClient)

    # Test embed_texts prepends document_prefix
    docs = ["First document.", "Second document."]
    model.embed_texts(docs)
    assert len(captured_inputs) == 1
    assert captured_inputs[0] == [
        "search_document: First document.",
        "search_document: Second document.",
    ]

    # Test embed_query prepends query_prefix
    captured_inputs.clear()
    model.embed_query("Who investigated Orion?")
    assert len(captured_inputs) == 1
    assert captured_inputs[0] == ["search_query: Who investigated Orion?"]


def test_openai_embedding_empty_prefixes_by_default(monkeypatch):
    """Test that OpenAIEmbeddingModel leaves texts untouched when prefixes are empty."""
    model = OpenAIEmbeddingModel(
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        dimension=128,
        query_prefix="",
        document_prefix="",
    )

    captured_inputs = []

    class MockResponse:
        def __init__(self, count):
            self.count = count

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": i, "embedding": [0.1] * 128} for i in range(self.count)]}

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            captured_inputs.append(json["input"])
            return MockResponse(len(json["input"]))

    monkeypatch.setattr("httpx.Client", MockClient)

    model.embed_texts(["Plain doc."])
    assert captured_inputs[0] == ["Plain doc."]

    captured_inputs.clear()
    model.embed_query("Plain query.")
    assert captured_inputs[0] == ["Plain query."]


def test_mock_embedding_prefixes_concatenation():
    """Test that MockEmbeddingModel correctly handles query and document prefixes."""
    model = MockEmbeddingModel(
        dimension=128,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    plain_model = MockEmbeddingModel(
        dimension=128,
        query_prefix="",
        document_prefix="",
    )

    # Document prefix test
    emb_doc = model.embed_texts(["Police report on suspect."])
    expected_doc_emb = plain_model.embed_texts(["search_document: Police report on suspect."])
    assert emb_doc == expected_doc_emb
    assert emb_doc != plain_model.embed_texts(["Police report on suspect."])

    # Query prefix test
    emb_query = model.embed_query("Suspect identity")
    expected_query_emb = plain_model.embed_query("search_query: Suspect identity")
    assert emb_query == expected_query_emb
    assert emb_query != plain_model.embed_query("Suspect identity")


def test_embedding_factory_with_prefixes():
    """Test that get_embedding_model forwards prefixes correctly."""
    m_mock = get_embedding_model(
        "mock",
        dimension=128,
        query_prefix="q_prefix: ",
        document_prefix="d_prefix: ",
    )
    assert isinstance(m_mock, MockEmbeddingModel)
    assert m_mock.query_prefix == "q_prefix: "
    assert m_mock.document_prefix == "d_prefix: "

    m_openai = get_embedding_model(
        "openai",
        model_name="nomic-embed",
        query_prefix="search_query: ",
        document_prefix="search_document: ",
    )
    assert isinstance(m_openai, OpenAIEmbeddingModel)
    assert m_openai.query_prefix == "search_query: "
    assert m_openai.document_prefix == "search_document: "


def test_openai_embedding_progress_callback(monkeypatch):
    model = OpenAIEmbeddingModel(
        model="text-embedding-3-small",
        base_url="https://api.openai.com/v1",
        api_key="sk-test",
        dimension=64,
        batch_size=2,
    )

    class MockResponse:
        def __init__(self, count):
            self.count = count

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": i, "embedding": [0.1] * 64} for i in range(self.count)]}

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, json, headers):
            return MockResponse(len(json["input"]))

    monkeypatch.setattr("httpx.Client", MockClient)

    calls = []

    def on_progress(completed, total):
        calls.append((completed, total))

    # 5 items with batch_size=2 should produce 3 callbacks: (2, 5), (4, 5), (5, 5)
    model.embed_texts(["t1", "t2", "t3", "t4", "t5"], progress_callback=on_progress)
    assert calls == [(2, 5), (4, 5), (5, 5)]


def test_mock_embedding_progress_callback():
    model = MockEmbeddingModel(dimension=64)
    calls = []

    def on_progress(completed, total):
        calls.append((completed, total))

    model.embed_texts(["a", "b", "c"], progress_callback=on_progress)
    assert calls == [(3, 3)]
