
from unittest.mock import MagicMock

import pytest

from app.services.retrieval_service import RetrievalService


# --------------------------------------------------
# Fixtures
# --------------------------------------------------

@pytest.fixture
def retrieval_dependencies():
    embedding_function = MagicMock(
        return_value=[0.1, 0.2, 0.3]
    )

    vector_store = MagicMock()

    vector_store.search.return_value = [
        {
            "chunk_id": "doc_001_source_0_chunk_0",
            "text": "Machine learning is a subset of AI.",
            "metadata": {"document_id": "doc_001"},
            "distance": 0.0,
        }
    ]

    document_registry = MagicMock()

    document_registry.get_completed_document_ids.return_value = [
        "doc_001",
        "doc_002",
    ]

    service = RetrievalService(
        embedding_function=embedding_function,
        vector_store=vector_store,
        document_registry=document_registry,
    )

    return (
        service,
        embedding_function,
        vector_store,
        document_registry,
    )


# --------------------------------------------------
# Successful Retrieval Tests
# --------------------------------------------------

def test_global_retrieval(retrieval_dependencies):
    """Verify global retrieval searches completed documents only."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    results = service.retrieve(
        query="What is machine learning?",
        top_k=5,
    )

    document_registry.get_completed_document_ids.assert_called_once_with()

    embedding_function.assert_called_once_with(
        "What is machine learning?"
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        document_ids=["doc_001", "doc_002"],
    )

    assert len(results) == 1
    assert results[0]["text"] == (
        "Machine learning is a subset of AI."
    )


def test_global_retrieval_uses_completed_document_ids(
    retrieval_dependencies,
):
    """Verify incomplete documents are excluded from global retrieval."""

    service, _, vector_store, document_registry = (
        retrieval_dependencies
    )

    document_registry.get_completed_document_ids.return_value = [
        "completed_001",
        "completed_002",
    ]

    service.retrieve(query="Explain artificial intelligence")

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
        document_ids=["completed_001", "completed_002"],
    )


def test_global_retrieval_returns_empty_when_no_completed_documents(
    retrieval_dependencies,
):
    """Verify global retrieval handles an empty completed-document list."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    document_registry.get_completed_document_ids.return_value = []

    results = service.retrieve(query="Explain AI")

    assert results == []

    document_registry.get_completed_document_ids.assert_called_once_with()
    embedding_function.assert_not_called()
    vector_store.search.assert_not_called()


def test_document_specific_retrieval(retrieval_dependencies):
    """Verify retrieval restricted to a completed document."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    document_registry.get_document.return_value = {
        "document_id": "doc_001",
        "status": "completed",
    }

    results = service.retrieve(
        query="Explain neural networks",
        top_k=3,
        document_id="doc_001",
    )

    document_registry.get_document.assert_called_once_with(
        "doc_001"
    )

    document_registry.get_completed_document_ids.assert_not_called()

    embedding_function.assert_called_once_with(
        "Explain neural networks"
    )

    vector_store.search.assert_called_once_with(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=3,
        document_id="doc_001",
    )

    assert len(results) == 1


def test_global_retrieval_does_not_check_specific_document(
    retrieval_dependencies,
):
    """Global retrieval should use completed IDs, not individual lookups."""

    service, _, _, document_registry = retrieval_dependencies

    service.retrieve(query="Explain AI")

    document_registry.get_document.assert_not_called()
    document_registry.get_completed_document_ids.assert_called_once_with()


# --------------------------------------------------
# Query Validation Tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "query",
    ["", " ", "\n", None, 123],
)
def test_retrieve_rejects_invalid_query(
    retrieval_dependencies,
    query,
):
    """Reject empty or invalid queries."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    with pytest.raises(ValueError, match="Query"):
        service.retrieve(query=query)

    embedding_function.assert_not_called()
    vector_store.search.assert_not_called()
    document_registry.get_completed_document_ids.assert_not_called()


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 1.5, True, "5", None],
)
def test_retrieve_rejects_invalid_top_k(
    retrieval_dependencies,
    top_k,
):
    """Reject invalid top_k values."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    with pytest.raises(ValueError, match="top_k"):
        service.retrieve(
            query="Explain machine learning",
            top_k=top_k,
        )

    embedding_function.assert_not_called()
    vector_store.search.assert_not_called()
    document_registry.get_completed_document_ids.assert_not_called()


# --------------------------------------------------
# Document Validation Tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "document_id",
    ["", " ", 123],
)
def test_retrieve_rejects_invalid_document_id(
    retrieval_dependencies,
    document_id,
):
    """Reject invalid document IDs."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    with pytest.raises(ValueError, match="document_id"):
        service.retrieve(
            query="Explain AI",
            document_id=document_id,
        )

    embedding_function.assert_not_called()
    vector_store.search.assert_not_called()
    document_registry.get_document.assert_not_called()


def test_retrieve_rejects_nonexistent_document(
    retrieval_dependencies,
):
    """Reject document IDs that do not exist."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    document_registry.get_document.return_value = None

    with pytest.raises(ValueError, match="Document not found"):
        service.retrieve(
            query="Explain AI",
            document_id="unknown_doc",
        )

    embedding_function.assert_not_called()
    vector_store.search.assert_not_called()


@pytest.mark.parametrize(
    "status",
    ["processing", "failed"],
)
def test_retrieve_rejects_unavailable_document(
    retrieval_dependencies,
    status,
):
    """Reject documents that have not completed ingestion."""

    service, embedding_function, vector_store, document_registry = (
        retrieval_dependencies
    )

    document_registry.get_document.return_value = {
        "document_id": "doc_001",
        "status": status,
    }

    with pytest.raises(
        ValueError,
        match="not available for retrieval",
    ):
        service.retrieve(
            query="Explain AI",
            document_id="doc_001",
        )

    embedding_function.assert_not_called()
    vector_store.search.assert_not_called()


# --------------------------------------------------
# Dependency Failure Tests
# --------------------------------------------------

def test_retrieve_propagates_embedding_failure(
    retrieval_dependencies,
):
    """Verify embedding errors are propagated."""

    service, embedding_function, vector_store, _ = (
        retrieval_dependencies
    )

    embedding_function.side_effect = RuntimeError(
        "Embedding model unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Embedding model unavailable",
    ):
        service.retrieve(query="Explain AI")

    vector_store.search.assert_not_called()


def test_retrieve_propagates_vector_store_failure(
    retrieval_dependencies,
):
    """Verify vector store errors are propagated."""

    service, _, vector_store, _ = retrieval_dependencies

    vector_store.search.side_effect = RuntimeError(
        "Vector database unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Vector database unavailable",
    ):
        service.retrieve(query="Explain AI")


# --------------------------------------------------
# Result Integrity Tests
# --------------------------------------------------

def test_retrieve_returns_empty_results(
    retrieval_dependencies,
):
    """Verify empty vector search results are returned correctly."""

    service, _, vector_store, _ = retrieval_dependencies

    vector_store.search.return_value = []

    results = service.retrieve(
        query="Information not present in documents"
    )

    assert results == []


def test_retrieve_preserves_vector_store_results(
    retrieval_dependencies,
):
    """Verify retrieved metadata and distances are preserved."""

    service, _, vector_store, _ = retrieval_dependencies

    expected_results = [
        {
            "chunk_id": "doc_001_source_0_chunk_0",
            "text": "Machine learning is a subset of AI.",
            "metadata": {
                "document_id": "doc_001",
                "page_number": 2,
                "chunk_index": 0,
            },
            "distance": 0.12,
        }
    ]

    vector_store.search.return_value = expected_results

    results = service.retrieve(
        query="What is machine learning?"
    )

    assert results == expected_results
    assert results[0]["metadata"]["page_number"] == 2
    assert results[0]["distance"] == 0.12
