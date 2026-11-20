
from langchain_core.documents import Document

from app.services.chunking import chunk_documents


def test_chunk_documents():
    document = Document(
        page_content=("Machine learning is a field of AI. " * 100),
        metadata={"source": "sample.pdf", "page": 1},
    )

    chunks = chunk_documents(
        [document],
        chunk_size=200,
        chunk_overlap=30,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert chunk.metadata["source"] == "sample.pdf"
        assert chunk.metadata["page"] == 1


def test_chunk_metadata():
    document = Document(
        page_content="This is a sample document. " * 30,
        metadata={"source": "sample.pdf"},
    )

    chunks = chunk_documents([document], chunk_size=100, chunk_overlap=20)

    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source_document_index"] == 0


def test_empty_document():
    document = Document(page_content="   ", metadata={})

    chunks = chunk_documents([document])

    assert chunks == []


def test_invalid_chunk_size():
    document = Document(page_content="Sample content")

    try:
        chunk_documents([document], chunk_size=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_invalid_overlap():
    document = Document(page_content="Sample content")

    try:
        chunk_documents([document], chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_heading_metadata_preserved():
    document = Document(
        page_content=(
            "# Motion Graphics\n\n"
            "## Opacity Adjustment\n\n"
            "Opacity changes based on distance.\n\n"
            "### Expression\n\n"
            "The expression calculates the final opacity."
        ),
        metadata={"source": "sample.pdf"},
    )

    chunks = chunk_documents([document])

    assert len(chunks) >= 1

    assert any(
        chunk.metadata.get("section")
        == "Motion Graphics > Opacity Adjustment > Expression"
        for chunk in chunks
    )


def test_code_block_preserved_when_it_fits():
    code = (
        "var p1 = thisLayer.transform.position;\n"
        'var p2 = thisComp.layer("Effector").transform.position;\n'
        "var d = length(p1, p2);\n"
        "d;"
    )

    document = Document(
        page_content=(
            "## Position Adjustment\n\n"
            "This expression adjusts position.\n\n"
            "```javascript\n"
            f"{code}\n"
            "```"
        ),
        metadata={"source": "sample.pdf"},
    )

    chunks = chunk_documents(
        [document],
        chunk_size=500,
        chunk_overlap=50,
    )

    assert any(
        code in chunk.page_content
        for chunk in chunks
    )


def test_chunk_indexes_are_sequential():
    document = Document(
        page_content=(
            "## Section One\n\n"
            + "First section content. " * 30
            + "\n\n## Section Two\n\n"
            + "Second section content. " * 30
        ),
        metadata={"source": "sample.pdf"},
    )

    chunks = chunk_documents(
        [document],
        chunk_size=150,
        chunk_overlap=30,
    )

    indexes = [
        chunk.metadata["chunk_index"]
        for chunk in chunks
    ]

    assert indexes == list(range(len(chunks)))
