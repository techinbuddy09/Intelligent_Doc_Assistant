
"""
Tests document upload and processing API.

Verifies successful uploads, duplicate detection,
and ingestion orchestration without invoking Docling,
embedding models, or actual vector storage.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import documents
from app.core.database import initialize_database
from app.services.document_registry import DocumentRegistry


client = TestClient(app)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Provide an isolated document registry for upload tests."""

    database_path = tmp_path / "test_documents.db"

    initialize_database(database_path)

    document_registry = DocumentRegistry(database_path)

    monkeypatch.setattr(
        app.state,
        "document_registry",
        document_registry,
        raising=False,
    )

    return document_registry


def mock_ingestion(pdf_path, document_id, vector_store):
    """Return mocked ingestion results for API testing."""

    return {
        "document_id": document_id,
        "documents_loaded": 1,
        "total_chunks": 5,
        "total_embeddings": 5,
        "stored_chunks": 5,
        "status": "completed",
    }


def test_document_upload(tmp_path, monkeypatch, registry):
    # Use temporary storage
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    # Provide a mock vector store through application state
    monkeypatch.setattr(
        app.state,
        "vector_store",
        object(),
        raising=False,
    )

    # Mock the ingestion pipeline
    monkeypatch.setattr(
        documents,
        "ingest_pdf",
        mock_ingestion,
    )

    pdf_content = b"%PDF-1.4\nSample PDF content"

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 201

    data = response.json()

    # Verify upload response
    assert data["filename"] == "sample.pdf"
    assert data["file_size"] == len(pdf_content)
    assert data["document_id"]

    # Verify ingestion results
    assert data["documents_loaded"] == 1
    assert data["total_chunks"] == 5
    assert data["total_embeddings"] == 5
    assert data["stored_chunks"] == 5
    assert data["status"] == "completed"

    # Verify document metadata in SQLite
    registered_document = registry.get_document(data["document_id"])

    assert registered_document is not None
    assert registered_document["status"] == "completed"
    assert registered_document["total_chunks"] == 5
    assert registered_document["filename"] == "sample.pdf"
    assert registered_document["file_size"] == len(pdf_content)

    # Verify file storage
    stored_file = tmp_path / f"{data['document_id']}.pdf"
    assert stored_file.exists()


def test_duplicate_document_upload(tmp_path, monkeypatch, registry):
    # Use temporary storage
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    # Provide a mock vector store
    monkeypatch.setattr(
        app.state,
        "vector_store",
        object(),
        raising=False,
    )

    # Mock ingestion
    monkeypatch.setattr(
        documents,
        "ingest_pdf",
        mock_ingestion,
    )

    pdf_content = b"%PDF-1.4\nSample PDF content"

    files = {
        "file": (
            "sample.pdf",
            pdf_content,
            "application/pdf",
        )
    }

    # First upload
    first_response = client.post(
        "/api/v1/documents/upload",
        files=files,
    )

    # Second upload
    second_response = client.post(
        "/api/v1/documents/upload",
        files=files,
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 200

    first_data = first_response.json()
    second_data = second_response.json()

    assert first_data["status"] == "completed"
    assert second_data["status"] == "duplicate"

    assert (
        first_data["document_id"]
        == second_data["document_id"]
    )

    # Verify only one document exists in SQLite
    registered_documents = registry.list_documents()

    assert len(registered_documents) == 1
    assert registered_documents[0]["document_id"] == first_data["document_id"]
    assert registered_documents[0]["status"] == "completed"

    # Verify only one PDF exists
    assert len(list(tmp_path.glob("*.pdf"))) == 1


def test_upload_rejects_invalid_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.txt",
                b"Some text",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are supported."


def test_upload_rejects_invalid_pdf_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.pdf",
                b"This is not a PDF",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF file."

    assert list(tmp_path.glob("*.pdf")) == []


def test_upload_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(documents, "MAX_FILE_SIZE", 10)

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "large.pdf",
                b"%PDF-1.4\nThis file is too large",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 413
    assert list(tmp_path.glob("*.pdf")) == []


def test_upload_handles_ingestion_failure(
    tmp_path,
    monkeypatch,
    registry,
):
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    monkeypatch.setattr(
        app.state,
        "vector_store",
        object(),
        raising=False,
    )

    def raise_ingestion_error(*args, **kwargs):
        raise RuntimeError("Unexpected ingestion failure")

    monkeypatch.setattr(
        documents,
        "ingest_pdf",
        raise_ingestion_error,
    )

    pdf_content = b"%PDF-1.4\nSample PDF content"

    response = client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "sample.pdf",
                pdf_content,
                "application/pdf",
            )
        },
    )

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "An unexpected error occurred during document ingestion."
    )

    # Failed ingestion must not leave an uploaded PDF behind.
    assert list(tmp_path.glob("*.pdf")) == []

    # Verify failed ingestion was recorded in SQLite.
    file_hash = documents.calculate_file_hash(pdf_content)

    failed_document = registry.get_document_by_hash(file_hash)

    assert failed_document is not None
    assert failed_document["status"] == "failed"


def test_failed_document_upload_can_be_retried(
    tmp_path,
    monkeypatch,
    registry,
):
    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    class MockVectorStore:
        def __init__(self):
            self.deleted_documents = []

        def delete_document(self, document_id):
            self.deleted_documents.append(document_id)

    vector_store = MockVectorStore()

    monkeypatch.setattr(
        app.state,
        "vector_store",
        vector_store,
        raising=False,
    )

    attempts = {"count": 0}

    def mock_retry_ingestion(pdf_path, document_id, vector_store):
        attempts["count"] += 1

        if attempts["count"] == 1:
            raise RuntimeError("Temporary ingestion failure")

        return {
            "document_id": document_id,
            "documents_loaded": 1,
            "total_chunks": 5,
            "total_embeddings": 5,
            "stored_chunks": 5,
            "status": "completed",
        }

    monkeypatch.setattr(
        documents,
        "ingest_pdf",
        mock_retry_ingestion,
    )

    pdf_content = b"%PDF-1.4\nRetry test content"

    def upload():
        return client.post(
            "/api/v1/documents/upload",
            files={
                "file": (
                    "retry.pdf",
                    pdf_content,
                    "application/pdf",
                )
            },
        )

    first_response = upload()

    assert first_response.status_code == 500

    file_hash = documents.calculate_file_hash(pdf_content)
    failed_document = registry.get_document_by_hash(file_hash)

    assert failed_document["status"] == "failed"

    original_document_id = failed_document["document_id"]

    second_response = upload()

    assert second_response.status_code == 201

    retry_data = second_response.json()

    assert retry_data["document_id"] == original_document_id
    assert retry_data["status"] == "completed"
    assert attempts["count"] == 2

    completed_document = registry.get_document(original_document_id)

    assert completed_document["status"] == "completed"
    assert completed_document["total_chunks"] == 5

    assert vector_store.deleted_documents == [original_document_id]


def test_list_documents(registry):
    document = registry.register_document(
        document_id="test-document-1",
        filename="sample.pdf",
        file_hash="hash-1",
        file_size=1024,
        status="completed",
    )

    response = client.get("/api/v1/documents")

    assert response.status_code == 200

    data = response.json()

    assert len(data["documents"]) == 1
    assert data["documents"][0]["document_id"] == document["document_id"]
    assert data["documents"][0]["filename"] == "sample.pdf"
    assert data["documents"][0]["status"] == "completed"


def test_delete_document(tmp_path, monkeypatch, registry):
    document_id = "test-document-2"

    registry.register_document(
        document_id=document_id,
        filename="sample.pdf",
        file_hash="hash-2",
        file_size=1024,
        status="completed",
    )

    monkeypatch.setattr(documents, "UPLOAD_DIR", tmp_path)

    pdf_path = tmp_path / f"{document_id}.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    class MockVectorStore:
        def delete_document(self, document_id):
            return 5

    monkeypatch.setattr(
        app.state,
        "vector_store",
        MockVectorStore(),
        raising=False,
    )

    response = client.delete(
        f"/api/v1/documents/{document_id}"
    )

    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == 5

    assert registry.get_document(document_id) is None
    assert not pdf_path.exists()


def test_delete_nonexistent_document(registry):
    response = client.delete(
        "/api/v1/documents/nonexistent"
    )

    assert response.status_code == 404


def test_delete_processing_document(registry):
    document_id = "processing-document"

    registry.register_document(
        document_id=document_id,
        filename="processing.pdf",
        file_hash="hash-3",
        file_size=1024,
        status="processing",
    )

    response = client.delete(
        f"/api/v1/documents/{document_id}"
    )

    assert response.status_code == 409
    assert registry.get_document(document_id) is not None

