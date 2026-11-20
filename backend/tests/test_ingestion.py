
"""
Tests for the document ingestion service.
"""

from unittest.mock import Mock

import pytest
from langchain_core.documents import Document

from app.services import ingestion


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Sample extracted document.")
    ]


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="Sample chunk.",
            metadata={
                "chunk_index": 0,
                "source_document_index": 0,
            },
        )
    ]


@pytest.fixture
def mock_vector_store():
    store = Mock()
    store.add_chunks.return_value = 1
    return store


def test_ingest_pdf_success(
    monkeypatch,
    sample_documents,
    sample_chunks,
    mock_vector_store,
):
    """Verify the complete ingestion workflow."""

    mock_embeddings = [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        lambda path: sample_documents,
    )

    monkeypatch.setattr(
        ingestion,
        "chunk_documents",
        lambda documents: sample_chunks,
    )

    monkeypatch.setattr(
        ingestion,
        "embed_documents",
        lambda chunks: mock_embeddings,
    )

    result = ingestion.ingest_pdf(
        pdf_path="sample.pdf",
        document_id="doc_001",
        vector_store=mock_vector_store,
    )

    assert result["document_id"] == "doc_001"
    assert result["documents_loaded"] == 1
    assert result["total_chunks"] == 1
    assert result["total_embeddings"] == 1
    assert result["stored_chunks"] == 1
    assert result["status"] == "completed"

    mock_vector_store.add_chunks.assert_called_once_with(
        chunks=sample_chunks,
        embeddings=mock_embeddings,
        document_id="doc_001",
    )


def test_ingest_pdf_rejects_invalid_document_id(
    mock_vector_store,
):
    """Verify that invalid document IDs are rejected."""

    with pytest.raises(ValueError, match="document_id"):
        ingestion.ingest_pdf(
            pdf_path="sample.pdf",
            document_id=" ",
            vector_store=mock_vector_store,
        )


def test_ingest_pdf_rejects_empty_chunks(
    monkeypatch,
    sample_documents,
    mock_vector_store,
):
    """Verify that ingestion stops when no chunks are generated."""

    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        lambda path: sample_documents,
    )

    monkeypatch.setattr(
        ingestion,
        "chunk_documents",
        lambda documents: [],
    )

    with pytest.raises(ValueError, match="No text chunks"):
        ingestion.ingest_pdf(
            pdf_path="sample.pdf",
            document_id="doc_001",
            vector_store=mock_vector_store,
        )

    mock_vector_store.add_chunks.assert_not_called()


def test_ingest_pdf_propagates_loading_error(
    monkeypatch,
    mock_vector_store,
):
    """Verify that document loading errors are propagated."""

    def raise_loading_error(path):
        raise FileNotFoundError("PDF not found")

    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        raise_loading_error,
    )

    with pytest.raises(FileNotFoundError, match="PDF not found"):
        ingestion.ingest_pdf(
            pdf_path="missing.pdf",
            document_id="doc_001",
            vector_store=mock_vector_store,
        )

    mock_vector_store.add_chunks.assert_not_called()


def test_ingest_pdf_propagates_embedding_error(
    monkeypatch,
    sample_documents,
    sample_chunks,
    mock_vector_store,
):
    def raise_embedding_error(chunks):
        raise RuntimeError("Embedding generation failed")

    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        lambda path: sample_documents,
    )

    monkeypatch.setattr(
        ingestion,
        "chunk_documents",
        lambda documents: sample_chunks,
    )

    monkeypatch.setattr(
        ingestion,
        "embed_documents",
        raise_embedding_error,
    )

    with pytest.raises(RuntimeError, match="Embedding generation failed"):
        ingestion.ingest_pdf(
            pdf_path="sample.pdf",
            document_id="doc_001",
            vector_store=mock_vector_store,
        )

    mock_vector_store.add_chunks.assert_not_called()


def test_ingest_pdf_propagates_storage_error(
    monkeypatch,
    sample_documents,
    sample_chunks,
    mock_vector_store,
):
    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        lambda path: sample_documents,
    )

    monkeypatch.setattr(
        ingestion,
        "chunk_documents",
        lambda documents: sample_chunks,
    )

    monkeypatch.setattr(
        ingestion,
        "embed_documents",
        lambda chunks: [[0.1, 0.2, 0.3]],
    )

    mock_vector_store.add_chunks.side_effect = RuntimeError(
        "Vector storage failed"
    )

    with pytest.raises(RuntimeError, match="Vector storage failed"):
        ingestion.ingest_pdf(
            pdf_path="sample.pdf",
            document_id="doc_001",
            vector_store=mock_vector_store,
        )


def test_ingest_pdf_handles_empty_extraction(
    monkeypatch,
    mock_vector_store,
):
    monkeypatch.setattr(
        ingestion,
        "load_pdf_documents",
        lambda path: [],
    )

    monkeypatch.setattr(
        ingestion,
        "chunk_documents",
        lambda documents: [],
    )

    with pytest.raises(ValueError, match="No text chunks"):
        ingestion.ingest_pdf(
            pdf_path="sample.pdf",
            document_id="doc_001",
            vector_store=mock_vector_store,
        )

    mock_vector_store.add_chunks.assert_not_called()
