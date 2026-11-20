
import httpx
import pytest

from unittest.mock import Mock

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


@pytest.fixture
def rag_service(monkeypatch):
    """Provide a mocked RAG service."""

    service = Mock()

    service.answer_question.return_value = {
        "question": "What is machine learning?",
        "answer": "Machine learning is a branch of artificial intelligence.",
        "sources": [
            {
                "document_id": "doc-123",
                "filename": "machine_learning.pdf",
                "chunk_ids": ["chunk-1", "chunk-2"],
            }
        ],
    }

    monkeypatch.setattr(
        app.state,
        "rag_service",
        service,
        raising=False,
    )

    return service


# --------------------------------------------------
# Successful question-answering tests
# --------------------------------------------------


def test_answer_question_success(rag_service):
    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == "What is machine learning?"

    assert data["answer"] == (
        "Machine learning is a branch of artificial intelligence."
    )

    assert len(data["sources"]) == 1

    assert data["sources"][0]["document_id"] == "doc-123"
    assert data["sources"][0]["filename"] == "machine_learning.pdf"
    assert data["sources"][0]["chunk_ids"] == [
        "chunk-1",
        "chunk-2",
    ]

    rag_service.answer_question.assert_called_once_with(
        question="What is machine learning?",
        document_id=None,
    )


def test_answer_question_with_document_id(rag_service):
    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
            "document_id": "doc-123",
        },
    )

    assert response.status_code == 200

    rag_service.answer_question.assert_called_once_with(
        question="What is machine learning?",
        document_id="doc-123",
    )


def test_answer_question_trims_whitespace(rag_service):
    response = client.post(
        "/api/v1/questions",
        json={
            "question": "  What is machine learning?  ",
        },
    )

    assert response.status_code == 200

    rag_service.answer_question.assert_called_once_with(
        question="What is machine learning?",
        document_id=None,
    )


# --------------------------------------------------
# Request validation tests
# --------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\n",
    ],
)
def test_rejects_empty_question(rag_service, question):
    response = client.post(
        "/api/v1/questions",
        json={
            "question": question,
        },
    )

    assert response.status_code == 422

    rag_service.answer_question.assert_not_called()


def test_rejects_missing_question(rag_service):
    response = client.post(
        "/api/v1/questions",
        json={},
    )

    assert response.status_code == 422

    rag_service.answer_question.assert_not_called()


def test_rejects_question_exceeding_maximum_length(rag_service):
    response = client.post(
        "/api/v1/questions",
        json={
            "question": "a" * 4001,
        },
    )

    assert response.status_code == 422

    rag_service.answer_question.assert_not_called()


def test_rejects_empty_document_id(rag_service):
    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
            "document_id": "   ",
        },
    )

    assert response.status_code == 422

    rag_service.answer_question.assert_not_called()


# --------------------------------------------------
# Document error tests
# --------------------------------------------------


def test_returns_404_when_document_does_not_exist(rag_service):
    rag_service.answer_question.side_effect = ValueError(
        "Document not found."
    )

    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
            "document_id": "invalid-document",
        },
    )

    assert response.status_code == 404

    assert response.json()["detail"] == "Document not found."


def test_returns_409_when_document_is_not_completed(rag_service):
    rag_service.answer_question.side_effect = ValueError(
        "Document is not available for retrieval."
    )

    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
            "document_id": "processing-document",
        },
    )

    assert response.status_code == 409

    assert response.json()["detail"] == (
        "Document is not available for retrieval."
    )


# --------------------------------------------------
# Ollama availability tests
# --------------------------------------------------


def test_returns_503_when_ollama_is_unavailable(rag_service):
    rag_service.answer_question.side_effect = httpx.ConnectError(
        "Ollama unavailable"
    )

    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
        },
    )

    assert response.status_code == 503

    assert response.json()["detail"] == (
        "The language model service is currently unavailable."
    )


# --------------------------------------------------
# Unexpected error tests
# --------------------------------------------------


def test_returns_500_for_unexpected_errors(rag_service):
    rag_service.answer_question.side_effect = RuntimeError(
        "Unexpected internal failure"
    )

    response = client.post(
        "/api/v1/questions",
        json={
            "question": "What is machine learning?",
        },
    )

    assert response.status_code == 500

    assert response.json()["detail"] == (
        "An unexpected error occurred while answering the question."
    )
