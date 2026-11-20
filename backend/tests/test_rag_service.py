
from unittest.mock import Mock

import pytest

from app.services.rag_service import RAGService


@pytest.fixture
def retrieval_service():
    return Mock()


@pytest.fixture
def document_registry():
    registry = Mock()
    registry.get_document.side_effect = lambda document_id: {
        "document_id": document_id,
        "filename": f"{document_id}.pdf",
        "status": "completed",
    }
    return registry


@pytest.fixture
def llm():
    return Mock()


@pytest.fixture
def rag_service(retrieval_service, document_registry, llm):
    return RAGService(
        retrieval_service=retrieval_service,
        document_registry=document_registry,
        llm=llm,
        top_k=5,
    )


def make_chunk(
    document_id="doc-1",
    chunk_id="chunk-1",
    text="Machine learning is a branch of artificial intelligence.",
):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "document_id": document_id,
            "chunk_index": 0,
            "section": "Introduction",
        },
        "distance": 0.25,
    }


# --------------------------------------------------
# Initialization tests
# --------------------------------------------------


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "5"])
def test_invalid_top_k_raises_error(
    retrieval_service,
    document_registry,
    llm,
    top_k,
):
    with pytest.raises(ValueError, match="top_k"):
        RAGService(
            retrieval_service,
            document_registry,
            llm,
            top_k=top_k,
        )


def test_valid_initialization(
    retrieval_service,
    document_registry,
    llm,
):
    service = RAGService(
        retrieval_service,
        document_registry,
        llm,
        top_k=3,
    )

    assert service.retrieval_service is retrieval_service
    assert service.document_registry is document_registry
    assert service.llm is llm
    assert service.top_k == 3


# --------------------------------------------------
# Question validation tests
# --------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["", " ", "\n", None, 123],
)
def test_invalid_question_raises_error(rag_service, question):
    with pytest.raises(ValueError, match="Question"):
        rag_service.answer_question(question)


def test_question_is_stripped_before_retrieval(
    rag_service,
    retrieval_service,
):
    retrieval_service.retrieve.return_value = []

    rag_service.answer_question("  What is AI?  ")

    retrieval_service.retrieve.assert_called_once_with(
        query="What is AI?",
        top_k=5,
        document_id=None,
    )


# --------------------------------------------------
# Retrieval tests
# --------------------------------------------------


def test_retrieval_uses_configured_top_k(
    rag_service,
    retrieval_service,
):
    retrieval_service.retrieve.return_value = []

    rag_service.answer_question("What is AI?")

    retrieval_service.retrieve.assert_called_once_with(
        query="What is AI?",
        top_k=5,
        document_id=None,
    )


def test_document_specific_retrieval(
    rag_service,
    retrieval_service,
):
    retrieval_service.retrieve.return_value = []

    rag_service.answer_question(
        "What is AI?",
        document_id="doc-1",
    )

    retrieval_service.retrieve.assert_called_once_with(
        query="What is AI?",
        top_k=5,
        document_id="doc-1",
    )


# --------------------------------------------------
# Empty retrieval tests
# --------------------------------------------------


def test_empty_retrieval_returns_fallback_answer(
    rag_service,
    retrieval_service,
    llm,
):
    retrieval_service.retrieve.return_value = []

    result = rag_service.answer_question("What is AI?")

    assert result["question"] == "What is AI?"
    assert "couldn't find relevant information" in result["answer"]
    assert result["sources"] == []

    llm.invoke.assert_not_called()


# --------------------------------------------------
# Context construction tests
# --------------------------------------------------


def test_build_context_creates_source_reference(rag_service):
    chunks = [make_chunk()]

    context, sources = rag_service._build_context(chunks)

    assert "[Source 1]" in context
    assert "doc-1.pdf" in context
    assert "Machine learning" in context

    assert len(sources) == 1
    assert sources[0]["document_id"] == "doc-1"
    assert sources[0]["filename"] == "doc-1.pdf"
    assert sources[0]["chunk_ids"] == ["chunk-1"]


def test_multiple_chunks_from_same_document_share_source(
    rag_service,
):
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            text="First chunk.",
        ),
        make_chunk(
            chunk_id="chunk-2",
            text="Second chunk.",
        ),
    ]

    context, sources = rag_service._build_context(chunks)

    assert context.count("[Source 1]") == 2
    assert "[Source 2]" not in context

    assert len(sources) == 1
    assert sources[0]["chunk_ids"] == ["chunk-1", "chunk-2"]


def test_chunks_from_different_documents_get_different_sources(
    rag_service,
):
    chunks = [
        make_chunk(
            document_id="doc-1",
            chunk_id="chunk-1",
            text="First document.",
        ),
        make_chunk(
            document_id="doc-2",
            chunk_id="chunk-2",
            text="Second document.",
        ),
    ]

    context, sources = rag_service._build_context(chunks)

    assert "[Source 1]" in context
    assert "[Source 2]" in context

    assert len(sources) == 2
    assert sources[0]["document_id"] == "doc-1"
    assert sources[1]["document_id"] == "doc-2"



def test_missing_registry_document_does_not_invent_filename(
    rag_service,
    document_registry,
):
    document_registry.get_document.side_effect = None
    document_registry.get_document.return_value = None

    context, sources = rag_service._build_context(
        [make_chunk()]
    )

    assert "Unknown document" in context
    assert sources[0]["filename"] is None



def test_missing_document_id_raises_error(rag_service):
    chunk = make_chunk()
    chunk["metadata"] = {}

    with pytest.raises(ValueError, match="document_id"):
        rag_service._build_context([chunk])


def test_non_string_chunk_text_raises_error(rag_service):
    chunk = make_chunk()
    chunk["text"] = None

    with pytest.raises(ValueError, match="text"):
        rag_service._build_context([chunk])


# --------------------------------------------------
# Prompt tests
# --------------------------------------------------


def test_prompt_contains_question_and_context():
    prompt = RAGService._build_prompt(
        question="What is machine learning?",
        context="Machine learning is a branch of AI.",
    )

    assert "What is machine learning?" in prompt
    assert "Machine learning is a branch of AI." in prompt


def test_prompt_restricts_answer_to_context():
    prompt = RAGService._build_prompt(
        question="What is AI?",
        context="AI is artificial intelligence.",
    )

    assert "only the information" in prompt
    assert "Do not invent facts" in prompt
    assert "untrusted reference material" in prompt


# --------------------------------------------------
# LLM response tests
# --------------------------------------------------


def test_extract_response_from_string():
    result = RAGService._extract_response_text("Hello")

    assert result == "Hello"


def test_extract_response_from_langchain_message():
    message = Mock()
    message.content = "Machine learning is a field of AI."

    result = RAGService._extract_response_text(message)

    assert result == "Machine learning is a field of AI."


def test_extract_response_from_content_blocks():
    message = Mock()
    message.content = [
        {"type": "text", "text": "First part."},
        {"type": "text", "text": "Second part."},
    ]

    result = RAGService._extract_response_text(message)

    assert result == "First part.\nSecond part."


def test_unsupported_llm_response_raises_error():
    with pytest.raises(RuntimeError, match="unsupported"):
        RAGService._extract_response_text(object())


def test_empty_llm_response_raises_error(
    rag_service,
    retrieval_service,
    llm,
):
    retrieval_service.retrieve.return_value = [make_chunk()]

    llm.invoke.return_value = Mock(content="   ")

    with pytest.raises(RuntimeError, match="empty answer"):
        rag_service.answer_question("What is AI?")


# --------------------------------------------------
# End-to-end service tests with mocked dependencies
# --------------------------------------------------


def test_answer_question_returns_answer_and_sources(
    rag_service,
    retrieval_service,
    llm,
):
    retrieval_service.retrieve.return_value = [
        make_chunk()
    ]

    llm.invoke.return_value = Mock(
        content="Machine learning is a branch of AI."
    )

    result = rag_service.answer_question(
        "What is machine learning?"
    )

    assert result["question"] == "What is machine learning?"
    assert result["answer"] == "Machine learning is a branch of AI."

    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "doc-1"
    assert result["sources"][0]["filename"] == "doc-1.pdf"

    llm.invoke.assert_called_once()


def test_llm_receives_retrieved_context(
    rag_service,
    retrieval_service,
    llm,
):
    retrieval_service.retrieve.return_value = [
        make_chunk(text="Neural networks contain interconnected nodes.")
    ]

    llm.invoke.return_value = Mock(
        content="Neural networks contain interconnected nodes."
    )

    rag_service.answer_question("What are neural networks?")

    prompt = llm.invoke.call_args.args[0]

    assert "Neural networks contain interconnected nodes." in prompt
    assert "What are neural networks?" in prompt


def test_llm_exception_is_propagated(
    rag_service,
    retrieval_service,
    llm,
):
    retrieval_service.retrieve.return_value = [make_chunk()]
    llm.invoke.side_effect = RuntimeError("Ollama unavailable")

    with pytest.raises(RuntimeError, match="Ollama unavailable"):
        rag_service.answer_question("What is AI?")
