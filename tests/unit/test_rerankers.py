"""Unit tests for Reranker implementations and factory."""

import httpx
import pytest

from bench_rag.core.chunker import Chunk
from bench_rag.core.vector_store import SearchResult
from bench_rag.rerankers import HttpReranker, MockReranker, get_reranker


def _create_sample_results() -> list[SearchResult]:
    chunks = [
        Chunk(
            id="c1",
            text="Portaria de abertura do inquérito policial 001/2023.",
            document_id="IP_001_2023",
            metadata={"page_number": 1},
        ),
        Chunk(
            id="c2",
            text="Transferência bancária suspeita da offshore Phoenix Capital para Carlos Eduardo.",
            document_id="IP_002_2023",
            metadata={"page_number": 12},
        ),
        Chunk(
            id="c3",
            text="Auto de apreensão de bens e documentos na residência do investigado.",
            document_id="IP_003_2023",
            metadata={"page_number": 5},
        ),
    ]
    return [
        SearchResult(chunk=chunks[0], score=0.85),
        SearchResult(chunk=chunks[1], score=0.80),
        SearchResult(chunk=chunks[2], score=0.75),
    ]


def test_get_reranker_factory():
    assert get_reranker("none") is None
    assert get_reranker(None) is None
    assert get_reranker("") is None

    mock = get_reranker("mock", model="custom-mock")
    assert isinstance(mock, MockReranker)
    assert mock.model_name == "custom-mock"

    for r_type in ["http", "vllm", "tei", "cohere", "jina"]:
        client = get_reranker(
            r_type, model="BAAI/bge-reranker-v2-m3", base_url="http://vllm:8000/v1"
        )
        assert isinstance(client, HttpReranker)
        assert client.model == "BAAI/bge-reranker-v2-m3"

    with pytest.raises(ValueError, match="Unknown reranker_type"):
        get_reranker("invalid_type")


def test_mock_reranker_reordering_and_metadata():
    reranker = MockReranker()
    results = _create_sample_results()

    query = "Carlos Eduardo transferiu recursos da offshore Phoenix Capital?"
    reranked = reranker.rerank(query=query, results=results, top_k=2)

    assert len(reranked) == 2
    # Chunk c2 has multiple overlapping tokens ('carlos', 'eduardo', 'offshore', 'phoenix', 'capital')
    # and should be promoted to rank 1 over c1
    top_chunk = reranked[0].chunk
    assert top_chunk.id == "c2"
    assert top_chunk.metadata["initial_retrieval_rank"] == 2
    assert top_chunk.metadata["initial_retrieval_score"] == 0.80
    assert reranked[0].score > 0.80


def test_mock_reranker_empty():
    reranker = MockReranker()
    assert reranker.rerank("qualquer query", [], top_k=5) == []


def test_http_reranker_endpoint_resolution():
    r1 = HttpReranker(base_url="http://host:8000")
    assert r1.endpoint == "http://host:8000/v1/rerank"

    r2 = HttpReranker(base_url="http://host:8000/v1")
    assert r2.endpoint == "http://host:8000/v1/rerank"

    r3 = HttpReranker(base_url="http://host:8000/rerank")
    assert r3.endpoint == "http://host:8000/rerank"


def test_http_reranker_success_payload_and_metadata(monkeypatch):
    results = _create_sample_results()

    def mock_post(url, json=None, headers=None):
        assert json["model"] == "BAAI/bge-reranker-v2-m3"
        assert json["query"] == "offshore carlos eduardo"
        assert len(json["documents"]) == 3
        assert json["top_n"] == 2

        response_data = {
            "id": "rerank-123",
            "results": [
                {"index": 1, "relevance_score": 0.985},
                {"index": 0, "relevance_score": 0.340},
            ],
        }
        return httpx.Response(200, json=response_data, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx.Client, "post", lambda self, url, **kwargs: mock_post(url, **kwargs))

    reranker = HttpReranker(
        model="BAAI/bge-reranker-v2-m3",
        base_url="http://vllm:8000/v1",
        api_key="secret-key",
    )
    reranked = reranker.rerank(query="offshore carlos eduardo", results=results, top_k=2)

    assert len(reranked) == 2
    assert reranked[0].chunk.id == "c2"
    assert reranked[0].score == 0.985
    assert reranked[0].chunk.metadata["initial_retrieval_rank"] == 2
    assert reranked[0].chunk.metadata["initial_retrieval_score"] == 0.80

    assert reranked[1].chunk.id == "c1"
    assert reranked[1].score == 0.340
    assert reranked[1].chunk.metadata["initial_retrieval_rank"] == 1


def test_http_reranker_fail_fast_on_error(monkeypatch):
    results = _create_sample_results()

    def mock_post_err(url, **kwargs):
        return httpx.Response(500, text="Internal Server Error", request=httpx.Request("POST", url))

    monkeypatch.setattr(
        httpx.Client, "post", lambda self, url, **kwargs: mock_post_err(url, **kwargs)
    )

    reranker = HttpReranker(base_url="http://vllm:8000/v1")
    with pytest.raises(httpx.HTTPStatusError):
        reranker.rerank(query="test query", results=results, top_k=2)
