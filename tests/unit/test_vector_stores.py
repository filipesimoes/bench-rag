"""Unit tests for vector stores."""

from unittest.mock import MagicMock

import pytest

from bench_rag.config import settings
from bench_rag.core.chunker import Chunk
from bench_rag.core.vector_store import DistanceMetric
from bench_rag.vector_stores import MemoryVectorStore, QdrantVectorStore, get_vector_store


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            id="c1",
            document_id="IP_001",
            text="Carlos Mendes nominee director",
            metadata={"page_number": 1},
        ),
        Chunk(
            id="c2",
            document_id="IP_002",
            text="Orion Logistics cash smurfing",
            metadata={"page_number": 2},
        ),
        Chunk(
            id="c3",
            document_id="IP_003",
            text="Eduardo Silveira wire transfer",
            metadata={"page_number": 1},
        ),
    ]


def test_memory_vector_store_cosine(sample_chunks: list[Chunk]):
    store = MemoryVectorStore()
    store.create_collection("test_col", dimension=3, metric=DistanceMetric.COSINE)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.add_chunks("test_col", sample_chunks, embeddings)

    # Search with vector aligned with c1
    results = store.search("test_col", query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert pytest.approx(results[0].score, rel=1e-3) == 1.0

    # Test get_all_chunks
    all_chunks = store.get_all_chunks("test_col")
    assert len(all_chunks) == 3


def test_memory_vector_store_filtering(sample_chunks: list[Chunk]):
    store = MemoryVectorStore()
    store.create_collection("test_col", dimension=3)
    embeddings = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
    store.add_chunks("test_col", sample_chunks, embeddings)

    filtered = store.search(
        "test_col",
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        filter_criteria={"document_id": "IP_002"},
    )
    assert len(filtered) == 1
    assert filtered[0].chunk.id == "c2"


def test_memory_vector_store_distance_metrics(sample_chunks: list[Chunk]):
    # Euclidean
    store_l2 = MemoryVectorStore()
    store_l2.create_collection("col_l2", dimension=3, metric=DistanceMetric.EUCLIDEAN)
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store_l2.add_chunks("col_l2", sample_chunks, embeddings)
    res_l2 = store_l2.search("col_l2", query_vector=[1.0, 0.0, 0.0], top_k=1)
    assert res_l2[0].chunk.id == "c1"


def test_qdrant_vector_store_mocked(sample_chunks: list[Chunk]):
    mock_client = MagicMock()
    mock_client.get_collections.return_value.collections = []

    store = QdrantVectorStore(client=mock_client)

    store.create_collection("col_q", dimension=3, metric=DistanceMetric.COSINE)
    assert mock_client.create_collection.called

    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.add_chunks("col_q", sample_chunks, embeddings)
    assert mock_client.upsert.called

    # Test query_points call when client has query_points
    mock_point = MagicMock()
    mock_point.id = "c1"
    mock_point.score = 0.95
    mock_point.payload = {
        "chunk_id": "c1",
        "document_id": "IP_001",
        "text": "test",
        "metadata": {},
        "start_char": 0,
        "end_char": 4,
        "token_count": 1,
    }
    mock_query_response = MagicMock()
    mock_query_response.points = [mock_point]
    mock_client.query_points.return_value = mock_query_response

    results = store.search("col_q", query_vector=[1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
    assert results[0].score == 0.95
    assert mock_client.query_points.called


def test_qdrant_vector_store_legacy_fallback(sample_chunks: list[Chunk]):
    mock_client = MagicMock(spec=["get_collections", "create_collection", "upsert", "search"])
    mock_client.get_collections.return_value.collections = []

    mock_hit = MagicMock()
    mock_hit.id = "c1"
    mock_hit.score = 0.88
    mock_hit.payload = {
        "chunk_id": "c1",
        "document_id": "IP_001",
        "text": "legacy",
        "metadata": {},
        "start_char": 0,
        "end_char": 6,
        "token_count": 1,
    }
    mock_client.search.return_value = [mock_hit]

    store = QdrantVectorStore(client=mock_client)
    assert not hasattr(store.client, "query_points")

    results = store.search("col_legacy", query_vector=[1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
    assert results[0].score == 0.88
    assert mock_client.search.called


def test_qdrant_vector_store_in_memory(sample_chunks: list[Chunk]):
    store = QdrantVectorStore(location=":memory:")
    store.create_collection("col_mem", dimension=3, metric=DistanceMetric.COSINE)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.add_chunks("col_mem", sample_chunks, embeddings)

    # 1. Search aligned with c1
    results = store.search("col_mem", query_vector=[1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "c1"
    assert pytest.approx(results[0].score, rel=1e-3) == 1.0

    # 2. Search with filter
    filtered = store.search(
        "col_mem",
        query_vector=[1.0, 0.0, 0.0],
        top_k=5,
        filter_criteria={"document_id": "IP_002"},
    )
    assert len(filtered) == 1
    assert filtered[0].chunk.id == "c2"

    # 3. get_all_chunks scrolling
    all_chunks = store.get_all_chunks("col_mem")
    assert len(all_chunks) == 3


def test_vector_store_factory():
    s1 = get_vector_store("memory")
    assert isinstance(s1, MemoryVectorStore)

    s2 = get_vector_store("qdrant")
    assert isinstance(s2, QdrantVectorStore)

    with pytest.raises(ValueError, match="Unsupported vector store type"):
        get_vector_store("unknown")


def test_vector_store_timeout_configuration():
    store = QdrantVectorStore(location=":memory:", timeout=120.0)
    assert store.timeout == 120.0

    store_default = QdrantVectorStore(location=":memory:")
    assert store_default.timeout == settings.qdrant_timeout

    store_factory = get_vector_store("qdrant", timeout=90.0)
    assert isinstance(store_factory, QdrantVectorStore)
    assert store_factory.timeout == 90.0


def test_vector_stores_add_chunks_progress_callback(sample_chunks: list[Chunk]):
    # Test memory store
    mem_store = MemoryVectorStore()
    embeddings = [[0.1, 0.2, 0.3]] * len(sample_chunks)
    mem_calls = []

    mem_store.add_chunks(
        "test_col",
        sample_chunks,
        embeddings,
        progress_callback=lambda c, t: mem_calls.append((c, t)),
    )
    assert mem_calls == [(len(sample_chunks), len(sample_chunks))]

    # Test Qdrant store with mock client
    mock_client = MagicMock()
    qdrant_store = QdrantVectorStore(client=mock_client)
    qdrant_calls = []

    qdrant_store.add_chunks(
        "test_col",
        sample_chunks,
        embeddings,
        progress_callback=lambda c, t: qdrant_calls.append((c, t)),
    )
    assert mock_client.upsert.called
    assert qdrant_calls == [(len(sample_chunks), len(sample_chunks))]
