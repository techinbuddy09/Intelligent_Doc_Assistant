
"""
Tests for the embedding generation service.

Tests cover:
    - Document embedding generation.
    - Query embedding generation.
    - Embedding dimensions.
    - Embedding normalization.
    - Input ordering.
    - Empty document handling.
    - Invalid query handling.
    - Model caching.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.services.embedding import (
    BATCH_SIZE,
    MODEL_NAME,
    NORMALIZE_EMBEDDINGS,
    QUERY_PREFIX,
    embed_documents,
    embed_query,
    get_embedding_model,
)


@pytest.fixture
def sample_documents():
    """Provide sample LangChain documents for embedding tests."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"chunk_index": 0},
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={"chunk_index": 1},
        ),
    ]


@pytest.fixture
def mock_embedding_model():
    """
    Mock SentenceTransformer to avoid loading the real model.

    Returns a model whose encode method generates predictable
    384-dimensional embeddings.
    """
    model = MagicMock()

    model.encode.side_effect = lambda texts, **kwargs: np.array(
        [
            np.full(384, index + 1, dtype=np.float32)
            for index, _ in enumerate(
                texts if isinstance(texts, list) else [texts]
            )
        ]
    )

    return model


def test_embedding_configuration():
    """Verify the selected embedding configuration."""
    assert MODEL_NAME == "BAAI/bge-small-en-v1.5"
    assert BATCH_SIZE == 32
    assert NORMALIZE_EMBEDDINGS is True
    assert QUERY_PREFIX.strip() != ""


@patch("app.services.embedding.get_embedding_model")
def test_embed_documents(mock_get_model, mock_embedding_model, sample_documents):
    """Verify that document chunks are converted into embeddings."""
    mock_get_model.return_value = mock_embedding_model

    embeddings = embed_documents(sample_documents)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    mock_embedding_model.encode.assert_called_once()

    call_args = mock_embedding_model.encode.call_args
    assert call_args.args[0] == [
        document.page_content for document in sample_documents
    ]


@patch("app.services.embedding.get_embedding_model")
def test_embed_documents_preserves_input_order(
    mock_get_model, mock_embedding_model, sample_documents
):
    """Verify that embedding order matches document order."""
    mock_get_model.return_value = mock_embedding_model

    embeddings = embed_documents(sample_documents)

    assert embeddings[0][0] == 1.0
    assert embeddings[1][0] == 2.0


@patch("app.services.embedding.get_embedding_model")
def test_embed_documents_uses_normalization(
    mock_get_model, mock_embedding_model, sample_documents
):
    """Verify that document embeddings are requested with normalization."""
    mock_get_model.return_value = mock_embedding_model

    embed_documents(sample_documents)

    kwargs = mock_embedding_model.encode.call_args.kwargs

    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert kwargs["batch_size"] == BATCH_SIZE


@patch("app.services.embedding.get_embedding_model")
def test_embed_documents_empty_input(mock_get_model):
    """Verify that empty document lists return an empty list."""
    embeddings = embed_documents([])

    assert embeddings == []
    mock_get_model.assert_not_called()


@patch("app.services.embedding.get_embedding_model")
def test_embed_query(mock_get_model, mock_embedding_model):
    """Verify that a user query is converted into an embedding."""
    mock_get_model.return_value = mock_embedding_model

    query = "What is machine learning?"

    embedding = embed_query(query)

    assert len(embedding) == 384
    assert all(isinstance(value, float) for value in embedding)

    mock_embedding_model.encode.assert_called_once()

    call_args = mock_embedding_model.encode.call_args

    assert call_args.args[0] == QUERY_PREFIX + query


@patch("app.services.embedding.get_embedding_model")
def test_embed_query_uses_normalization(
    mock_get_model, mock_embedding_model
):
    """Verify that query embeddings use normalization."""
    mock_get_model.return_value = mock_embedding_model

    embed_query("Explain neural networks")

    kwargs = mock_embedding_model.encode.call_args.kwargs

    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True


@pytest.mark.parametrize("query", ["", " ", "\n", "\t"])
def test_embed_query_rejects_empty_input(query):
    """Verify that empty or whitespace-only queries are rejected."""
    with pytest.raises(ValueError, match="Query cannot be empty"):
        embed_query(query)


def test_get_embedding_model_caching():
    """
    Verify that the embedding model is cached.

    Clear the cache before and after the test to avoid
    interference from other tests.
    """
    get_embedding_model.cache_clear()

    with patch(
        "app.services.embedding.SentenceTransformer"
    ) as mock_transformer, patch(
        "app.services.embedding.torch.cuda.is_available",
        return_value=False,
    ):
        first_model = get_embedding_model()
        second_model = get_embedding_model()

        assert first_model is second_model
        mock_transformer.assert_called_once_with(
            MODEL_NAME,
            device="cpu",
        )

    get_embedding_model.cache_clear()
