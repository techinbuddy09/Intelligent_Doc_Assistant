
"""
Tests for the document registry service.

Tests cover:
    - Document registration.
    - Duplicate document ID rejection.
    - Duplicate file hash rejection.
    - Document retrieval.
    - Nonexistent document retrieval.
    - Document listing.
    - Pagination.
    - Document status updates.
    - Invalid status rejection.
    - Document deletion.
    - Invalid input validation.
"""

import pytest

from app.core.database import initialize_database
from app.services.document_registry import DocumentRegistry


@pytest.fixture
def registry(tmp_path):
    """Provide an isolated document registry."""

    database_path = tmp_path / "test_documents.db"

    initialize_database(database_path)

    return DocumentRegistry(database_path)


@pytest.fixture
def sample_document(registry):
    """Register and return a sample document."""

    return registry.register_document(
        document_id="doc_001",
        filename="machine_learning.pdf",
        file_hash="hash_001",
        file_size=1024,
    )


def test_register_document(registry):
    """Verify successful document registration."""

    result = registry.register_document(
        document_id="doc_001",
        filename="machine_learning.pdf",
        file_hash="hash_001",
        file_size=1024,
    )

    assert result["document_id"] == "doc_001"
    assert result["filename"] == "machine_learning.pdf"
    assert result["file_hash"] == "hash_001"
    assert result["file_size"] == 1024
    assert result["status"] == "processing"
    assert result["total_chunks"] == 0
    assert result["created_at"] is not None


def test_register_document_rejects_duplicate_id(
    registry,
    sample_document,
):
    """Verify duplicate document IDs are rejected."""

    with pytest.raises(ValueError, match="already exists"):
        registry.register_document(
            document_id="doc_001",
            filename="another.pdf",
            file_hash="hash_002",
            file_size=2048,
        )


def test_register_document_rejects_duplicate_hash(
    registry,
    sample_document,
):
    """Verify duplicate file hashes are rejected."""

    with pytest.raises(ValueError, match="already exists"):
        registry.register_document(
            document_id="doc_002",
            filename="another.pdf",
            file_hash="hash_001",
            file_size=2048,
        )


def test_get_document(registry, sample_document):
    """Verify document retrieval by ID."""

    result = registry.get_document("doc_001")

    assert result is not None
    assert result["document_id"] == "doc_001"
    assert result["filename"] == "machine_learning.pdf"


def test_get_nonexistent_document(registry):
    """Verify missing documents return None."""

    result = registry.get_document("unknown_doc")

    assert result is None


def test_list_documents(registry):
    """Verify that all registered documents are returned."""

    registry.register_document(
        "doc_001", "first.pdf", "hash_001", 1024
    )

    registry.register_document(
        "doc_002", "second.pdf", "hash_002", 2048
    )

    results = registry.list_documents()

    assert len(results) == 2

    document_ids = {
        document["document_id"]
        for document in results
    }

    assert document_ids == {"doc_001", "doc_002"}


def test_list_documents_pagination(registry):
    """Verify document listing supports pagination."""

    for index in range(5):
        registry.register_document(
            document_id=f"doc_{index}",
            filename=f"document_{index}.pdf",
            file_hash=f"hash_{index}",
            file_size=1024,
        )

    first_page = registry.list_documents(
        limit=2,
        offset=0,
    )

    second_page = registry.list_documents(
        limit=2,
        offset=2,
    )

    assert len(first_page) == 2
    assert len(second_page) == 2

    first_ids = {
        document["document_id"]
        for document in first_page
    }

    second_ids = {
        document["document_id"]
        for document in second_page
    }

    assert first_ids.isdisjoint(second_ids)


@pytest.mark.parametrize(
    "limit, offset",
    [
        (0, 0),
        (-1, 0),
        (1.5, 0),
        (True, 0),
        (10, -1),
        (10, 1.5),
        (10, True),
    ],
)
def test_list_documents_rejects_invalid_pagination(
    registry,
    limit,
    offset,
):
    """Verify invalid pagination parameters are rejected."""

    with pytest.raises(ValueError):
        registry.list_documents(
            limit=limit,
            offset=offset,
        )


def test_update_document_status(registry, sample_document):
    """Verify document status and chunk count are updated."""

    result = registry.update_document_status(
        document_id="doc_001",
        status="completed",
        total_chunks=15,
    )

    assert result["status"] == "completed"
    assert result["total_chunks"] == 15


def test_update_document_status_without_chunk_count(
    registry,
    sample_document,
):
    """Verify status can be updated without changing chunk count."""

    registry.update_document_status(
        document_id="doc_001",
        status="completed",
        total_chunks=15,
    )

    result = registry.update_document_status(
        document_id="doc_001",
        status="failed",
    )

    assert result["status"] == "failed"
    assert result["total_chunks"] == 15


@pytest.mark.parametrize(
    "status",
    ["invalid", "", "pending", "processing123"],
)
def test_update_document_rejects_invalid_status(
    registry,
    sample_document,
    status,
):
    """Verify unsupported document statuses are rejected."""

    with pytest.raises(ValueError, match="status"):
        registry.update_document_status(
            document_id="doc_001",
            status=status,
        )


def test_update_nonexistent_document(registry):
    """Verify updating a missing document is rejected."""

    with pytest.raises(ValueError, match="not found"):
        registry.update_document_status(
            document_id="unknown_doc",
            status="completed",
        )


@pytest.mark.parametrize(
    "total_chunks",
    [-1, 1.5, True, "10"],
)
def test_update_document_rejects_invalid_chunk_count(
    registry,
    sample_document,
    total_chunks,
):
    """Verify invalid chunk counts are rejected."""

    with pytest.raises(ValueError, match="total_chunks"):
        registry.update_document_status(
            document_id="doc_001",
            status="completed",
            total_chunks=total_chunks,
        )


def test_delete_document(registry, sample_document):
    """Verify document metadata is deleted."""

    result = registry.delete_document("doc_001")

    assert result is True
    assert registry.get_document("doc_001") is None


def test_delete_nonexistent_document(registry):
    """Verify deleting a missing document returns False."""

    result = registry.delete_document("unknown_doc")

    assert result is False


@pytest.mark.parametrize(
    "document_id",
    ["", " ", None, 123],
)
def test_get_document_rejects_invalid_id(
    registry,
    document_id,
):
    """Verify invalid document IDs are rejected during retrieval."""

    with pytest.raises(ValueError, match="document_id"):
        registry.get_document(document_id)


@pytest.mark.parametrize(
    "document_id, filename, file_hash, file_size",
    [
        ("", "sample.pdf", "hash_001", 1024),
        ("doc_001", "", "hash_001", 1024),
        ("doc_001", "sample.pdf", "", 1024),
        ("doc_001", "sample.pdf", "hash_001", -1),
        ("doc_001", "sample.pdf", "hash_001", 1.5),
        ("doc_001", "sample.pdf", "hash_001", True),
    ],
)
def test_register_document_rejects_invalid_inputs(
    registry,
    document_id,
    filename,
    file_hash,
    file_size,
):
    """Verify invalid registration inputs are rejected."""

    with pytest.raises(ValueError):
        registry.register_document(
            document_id=document_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
        )


def test_register_document_rejects_invalid_status(registry):
    """Verify invalid initial document statuses are rejected."""

    with pytest.raises(ValueError, match="status"):
        registry.register_document(
            document_id="doc_001",
            filename="sample.pdf",
            file_hash="hash_001",
            file_size=1024,
            status="invalid",
        )


@pytest.mark.parametrize(
    "document_id",
    ["", " ", None, 123],
)
def test_delete_document_rejects_invalid_id(
    registry,
    document_id,
):
    """Verify invalid document IDs are rejected during deletion."""

    with pytest.raises(ValueError, match="document_id"):
        registry.delete_document(document_id)


def test_document_registry_persistence(tmp_path):
    """Verify records survive registry reinitialization."""

    database_path = tmp_path / "persistent_documents.db"

    initialize_database(database_path)

    first_registry = DocumentRegistry(database_path)

    first_registry.register_document(
        document_id="doc_001",
        filename="sample.pdf",
        file_hash="hash_001",
        file_size=1024,
    )

    second_registry = DocumentRegistry(database_path)

    result = second_registry.get_document("doc_001")

    assert result is not None
    assert result["filename"] == "sample.pdf"
    assert result["file_hash"] == "hash_001"


def test_get_document_by_hash_returns_existing_document(
    registry,
):
    document_id = registry.register_document(
        filename="report.pdf",
        file_hash="abc123",
        file_path="uploads/report.pdf",
        status="completed",
    )

    document = registry.get_document_by_hash("abc123")

    assert document is not None
    assert document["id"] == document_id
    assert document["filename"] == "report.pdf"
    assert document["file_hash"] == "abc123"
    assert document["status"] == "completed"



def test_get_document_by_hash_returns_existing_document(registry):
    registered_document = registry.register_document(
        document_id="doc_001",
        filename="report.pdf",
        file_hash="abc123",
        file_size=1024,
        status="completed",
    )

    document = registry.get_document_by_hash("abc123")

    assert document is not None
    assert document["document_id"] == registered_document["document_id"]
    assert document["filename"] == "report.pdf"
    assert document["file_hash"] == "abc123"
    assert document["status"] == "completed"


def test_get_completed_document_ids(registry):
    """Return only documents with completed status."""

    registry.register_document(
        document_id="doc_001",
        filename="machine_learning.pdf",
        file_hash="hash_001",
        file_size=1024,
        status="completed",
    )

    registry.register_document(
        document_id="doc_002",
        filename="deep_learning.pdf",
        file_hash="hash_002",
        file_size=2048,
        status="processing",
    )

    registry.register_document(
        document_id="doc_003",
        filename="neural_networks.pdf",
        file_hash="hash_003",
        file_size=3072,
        status="failed",
    )

    completed_ids = registry.get_completed_document_ids()

    assert completed_ids == ["doc_001"]


def test_get_completed_document_ids_multiple_completed(registry):
    """Return all completed document IDs."""

    registry.register_document(
        document_id="doc_001",
        filename="first.pdf",
        file_hash="hash_001",
        file_size=1024,
        status="completed",
    )

    registry.register_document(
        document_id="doc_002",
        filename="second.pdf",
        file_hash="hash_002",
        file_size=2048,
        status="completed",
    )

    registry.register_document(
        document_id="doc_003",
        filename="third.pdf",
        file_hash="hash_003",
        file_size=3072,
        status="processing",
    )

    completed_ids = registry.get_completed_document_ids()

    assert completed_ids == ["doc_001", "doc_002"]


def test_get_completed_document_ids_empty(registry):
    """Return an empty list when no completed documents exist."""

    completed_ids = registry.get_completed_document_ids()

    assert completed_ids == []

