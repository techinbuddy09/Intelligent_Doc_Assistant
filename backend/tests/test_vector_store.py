
"""
Tests for the ChromaDB vector storage service.

Tests cover:
    - Persistent vector store initialization.
    - Chunk insertion.
    - Vector similarity search.
    - Document filtering.
    - Document deletion.
    - Input validation.
"""

import pytest

from langchain_core.documents import Document

from app.services.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """
    Provide an isolated ChromaDB vector store for each test.

    tmp_path creates a unique temporary directory,
    preventing test data from interfering with other tests.
    """
    persist_directory = tmp_path / "chroma_db"

    return VectorStore(persist_directory=persist_directory)


@pytest.fixture
def sample_chunks():
    """Provide sample document chunks."""
    return [
        Document(
            page_content="Machine learning is a subset of AI.",
            metadata={"chunk_index": 0}
        ),
        Document(
            page_content="Deep learning uses neural networks.",
            metadata={"chunk_index": 1}
        ),
        Document(
            page_content="Supervised learning uses labelled data.",
            metadata={"chunk_index": 2}
        ),
    ]


@pytest.fixture
def sample_embeddings():
    """Provide predictable three-dimensional embeddings."""
    return [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ]


def test_add_chunks(vector_store, sample_chunks, sample_embeddings):
    """Verify that document chunks are stored successfully."""

    document_id = "doc_001"

    inserted_count = vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id=document_id,
    )

    assert inserted_count == 3

    # Verify actual records in ChromaDB.
    results = vector_store.collection.get(
        where={"document_id": document_id},
        include=["documents", "metadatas"],
    )

    assert len(results["ids"]) == 3
    assert len(results["documents"]) == 3

    for metadata in results["metadatas"]:
        assert metadata["document_id"] == document_id


def test_add_chunks_empty_input(vector_store):
    """Verify that empty chunks return zero."""

    inserted_count = vector_store.add_chunks(
        chunks=[],
        embeddings=[],
        document_id="doc_001",
    )

    assert inserted_count == 0


def test_add_chunks_mismatched_embeddings(
    vector_store, sample_chunks, sample_embeddings
):
    """Verify that mismatched chunk and embedding counts are rejected."""

    with pytest.raises(ValueError, match="Number of chunks"):
        vector_store.add_chunks(
            chunks=sample_chunks,
            embeddings=sample_embeddings[:2],
            document_id="doc_001",
        )

def test_search(vector_store, sample_chunks, sample_embeddings):
    """Verify that similarity search retrieves relevant chunks."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=2,
    )

    assert len(results) == 2

    # Exact same embedding should have zero cosine distance.
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)

    assert results[0]["text"] == sample_chunks[0].page_content
    assert results[0]["metadata"]["document_id"] == "doc_001"


def test_search_document_filter(
    vector_store, sample_chunks, sample_embeddings
):
    """Verify that search can be restricted to one document."""

    vector_store.add_chunks(
        chunks=sample_chunks[:2],
        embeddings=sample_embeddings[:2],
        document_id="doc_001",
    )

    vector_store.add_chunks(
        chunks=[
            Document(
                page_content="Artificial intelligence applications.",
                metadata={"chunk_index": 0},
            )
        ],
        embeddings=[sample_embeddings[2]],
        document_id="doc_002",
    )

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=5,
        document_id="doc_002",
    )

    assert len(results) == 1
    assert results[0]["metadata"]["document_id"] == "doc_002"


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_search_rejects_invalid_top_k(vector_store, top_k):
    """Verify that invalid top_k values are rejected."""

    with pytest.raises(ValueError, match="top_k"):
        vector_store.search(
            query_embedding=[0.1, 0.2, 0.3],
            top_k=top_k,
        )

def test_delete_document(vector_store, sample_chunks, sample_embeddings):
    """Verify that all chunks belonging to a document are deleted."""

    document_id = "doc_delete"

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id=document_id,
    )

    deleted_count = vector_store.delete_document(document_id)

    assert deleted_count == 3

    remaining = vector_store.collection.get(
        where={"document_id": document_id}
    )

    assert len(remaining["ids"]) == 0


def test_delete_nonexistent_document(vector_store):
    """Verify that deleting a nonexistent document returns zero."""

    deleted_count = vector_store.delete_document("unknown_doc")

    assert deleted_count == 0


def test_delete_document_repeatedly(
    vector_store, sample_chunks, sample_embeddings
):
    """Verify that repeated deletion is safe."""

    document_id = "doc_repeated"

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id=document_id,
    )

    assert vector_store.delete_document(document_id) == 3
    assert vector_store.delete_document(document_id) == 0

@pytest.mark.parametrize("document_id", ["", " ", None, 123])
def test_add_chunks_rejects_invalid_document_id(
    vector_store, sample_chunks, sample_embeddings, document_id
):
    with pytest.raises(ValueError):
        vector_store.add_chunks(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            document_id=document_id,
        )


@pytest.mark.parametrize("document_id", ["", " ", None, 123])
def test_delete_document_rejects_invalid_document_id(
    vector_store, document_id
):
    with pytest.raises(ValueError):
        vector_store.delete_document(document_id)

@pytest.mark.parametrize(
    "embedding",
    [
        [],
        [float("inf"), 0.2, 0.3],
        [float("nan"), 0.2, 0.3],
    ],
)
def test_search_rejects_invalid_embeddings(vector_store, embedding):
    with pytest.raises(ValueError):
        vector_store.search(query_embedding=embedding)


def test_add_chunks_rejects_empty_content(vector_store):
    """Verify that empty chunk content is rejected."""

    chunks = [
        Document(
            page_content="   ",
            metadata={"chunk_index": 0},
        )
    ]

    with pytest.raises(ValueError, match="Chunk content"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=[[0.1, 0.2, 0.3]],
            document_id="doc_001",
        )


def test_add_chunks_rejects_inconsistent_dimensions(vector_store):
    """Verify that embeddings must have matching dimensions."""

    chunks = [
        Document(
            page_content="First chunk",
            metadata={"chunk_index": 0}
        ),
        Document(
            page_content="Second chunk",
            metadata={"chunk_index": 1}
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5],
    ]

    with pytest.raises(ValueError, match="same dimensions"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id="doc_001",
        )


def test_add_chunks_rejects_invalid_chunk_index(vector_store):
    """Verify that negative chunk indices are rejected."""

    chunks = [
        Document(
            page_content="Machine learning",
            metadata={"chunk_index": -1},
        )
    ]

    with pytest.raises(ValueError, match="chunk_index"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=[[0.1, 0.2, 0.3]],
            document_id="doc_001",
        )


def test_search_rejects_empty_document_id(vector_store):
    """Verify that empty document filters are rejected."""

    with pytest.raises(ValueError, match="document_id"):
        vector_store.search(
            query_embedding=[0.1, 0.2, 0.3],
            document_id=" ",
        )

def test_vector_store_persistence(
    tmp_path, sample_chunks, sample_embeddings
):
    """Verify that stored vectors survive client reinitialization."""

    persist_directory = tmp_path / "persistent_chroma"

    # First instance: insert data.
    first_store = VectorStore(persist_directory=persist_directory)

    first_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_persistent",
    )

    # Second instance: initialize using the same directory.
    second_store = VectorStore(persist_directory=persist_directory)

    results = second_store.search(
        query_embedding=sample_embeddings[0],
        top_k=3,
        document_id="doc_persistent",
    )

    assert len(results) == 3
    assert results[0]["text"] == sample_chunks[0].page_content
    assert results[0]["metadata"]["document_id"] == "doc_persistent"


def test_add_chunks_rejects_duplicate_chunk_ids(vector_store):
    """Chunks must have unique generated IDs."""

    chunks = [
        Document(
            page_content="First chunk",
            metadata={"chunk_index": 0},
        ),
        Document(
            page_content="Second chunk",
            metadata={"chunk_index": 0},
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
    ]

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
            document_id="doc_001",
        )


def test_add_chunks_rejects_unsupported_metadata(vector_store):
    """Metadata must contain supported scalar values."""

    chunks = [
        Document(
            page_content="Machine learning",
            metadata={
                "chunk_index": 0,
                "tags": ["AI", "ML"],
            },
        )
    ]

    with pytest.raises(ValueError, match="Unsupported metadata"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=[[0.1, 0.2, 0.3]],
            document_id="doc_001",
        )


@pytest.mark.parametrize(
    "value",
    [float("inf"), float("-inf"), float("nan")],
)
def test_add_chunks_rejects_non_finite_metadata(
    vector_store, value
):
    """Non-finite metadata values must be rejected."""

    chunks = [
        Document(
            page_content="Machine learning",
            metadata={
                "chunk_index": 0,
                "score": value,
            },
        )
    ]

    with pytest.raises(ValueError, match="finite"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=[[0.1, 0.2, 0.3]],
            document_id="doc_001",
        )


@pytest.mark.parametrize(
    "embedding",
    [
        [],
        [float("inf"), 0.2, 0.3],
        [float("nan"), 0.2, 0.3],
    ],
)
def test_add_chunks_rejects_invalid_embeddings(
    vector_store, embedding
):
    """Invalid stored embeddings must be rejected."""

    chunks = [
        Document(
            page_content="Machine learning",
            metadata={"chunk_index": 0},
        )
    ]

    with pytest.raises(ValueError):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=[embedding],
            document_id="doc_001",
        )


@pytest.mark.parametrize(
    "source_index",
    [-1, 1.5, True, "0"],
)
def test_add_chunks_rejects_invalid_source_index(
    vector_store, source_index
):
    """Source document index must be a non-negative integer."""

    chunks = [
        Document(
            page_content="Machine learning",
            metadata={
                "chunk_index": 0,
                "source_document_index": source_index,
            },
        )
    ]

    with pytest.raises(ValueError, match="source_document_index"):
        vector_store.add_chunks(
            chunks=chunks,
            embeddings=[[0.1, 0.2, 0.3]],
            document_id="doc_001",
        )


def test_add_chunks_upsert_updates_existing_chunk(
    vector_store,
):
    """Upsert should update an existing chunk ID."""

    original_chunk = Document(
        page_content="Original content",
        metadata={"chunk_index": 0},
    )

    updated_chunk = Document(
        page_content="Updated content",
        metadata={"chunk_index": 0},
    )

    embedding = [[0.1, 0.2, 0.3]]

    vector_store.add_chunks(
        chunks=[original_chunk],
        embeddings=embedding,
        document_id="doc_001",
    )

    vector_store.add_chunks(
        chunks=[updated_chunk],
        embeddings=embedding,
        document_id="doc_001",
    )

    results = vector_store.collection.get(
        where={"document_id": "doc_001"},
        include=["documents"],
    )

    assert len(results["ids"]) == 1
    assert results["documents"][0] == "Updated content"


def test_delete_document_does_not_affect_other_documents(
    vector_store, sample_chunks, sample_embeddings
):
    """Deleting one document must preserve unrelated documents."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    vector_store.add_chunks(
        chunks=sample_chunks[:1],
        embeddings=sample_embeddings[:1],
        document_id="doc_002",
    )

    assert vector_store.delete_document("doc_001") == 3

    remaining = vector_store.collection.get(
        where={"document_id": "doc_002"},
        include=["documents"],
    )

    assert len(remaining["ids"]) == 1
    assert remaining["documents"][0] == sample_chunks[0].page_content

def test_search_empty_collection(vector_store):
    """Searching an empty collection should return an empty list."""

    results = vector_store.search(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=5,
    )

    assert results == []


def test_search_nonexistent_document(
    vector_store, sample_embeddings
):
    """Searching a nonexistent document should return no results."""

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=5,
        document_id="unknown_document",
    )

    assert results == []


def test_search_top_k_greater_than_available_chunks(
    vector_store, sample_chunks, sample_embeddings
):
    """Search should return all available chunks when top_k is larger."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=10,
    )

    assert len(results) == 3


def test_search_global_across_multiple_documents(
    vector_store, sample_chunks, sample_embeddings
):
    """Global search should retrieve chunks from multiple documents."""

    vector_store.add_chunks(
        chunks=sample_chunks[:2],
        embeddings=sample_embeddings[:2],
        document_id="doc_001",
    )

    vector_store.add_chunks(
        chunks=sample_chunks[2:],
        embeddings=sample_embeddings[2:],
        document_id="doc_002",
    )

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=5,
    )

    document_ids = {
        result["metadata"]["document_id"]
        for result in results
    }

    assert document_ids == {"doc_001", "doc_002"}


def test_search_results_are_ranked_by_distance(
    vector_store, sample_chunks, sample_embeddings
):
    """Results should be ordered from lowest to highest distance."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=3,
    )

    distances = [result["distance"] for result in results]

    assert distances == sorted(distances)

def test_add_chunks_rejects_existing_dimension_mismatch(
    vector_store, sample_chunks, sample_embeddings
):
    """Insertion must match the existing collection dimension."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        vector_store.add_chunks(
            chunks=[
                Document(
                    page_content="Another document",
                    metadata={"chunk_index": 0},
                )
            ],
            embeddings=[[0.1, 0.2]],
            document_id="doc_002",
        )


def test_search_rejects_query_dimension_mismatch(
    vector_store, sample_chunks, sample_embeddings
):
    """Query embeddings must match stored embedding dimensions."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        vector_store.search(
            query_embedding=[0.1, 0.2],
        )

def test_search_multiple_document_ids(
    vector_store, sample_chunks, sample_embeddings
):
    """Search should retrieve chunks from specified documents."""

    vector_store.add_chunks(
        chunks=sample_chunks[:2],
        embeddings=sample_embeddings[:2],
        document_id="doc_001",
    )

    vector_store.add_chunks(
        chunks=sample_chunks[2:],
        embeddings=sample_embeddings[2:],
        document_id="doc_002",
    )

    vector_store.add_chunks(
        chunks=sample_chunks[:1],
        embeddings=sample_embeddings[:1],
        document_id="doc_003",
    )

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        top_k=5,
        document_ids=["doc_001", "doc_002"],
    )

    document_ids = {
        result["metadata"]["document_id"]
        for result in results
    }

    assert document_ids == {"doc_001", "doc_002"}


def test_search_empty_document_ids(vector_store, sample_embeddings):
    """An empty document ID list should return no results."""

    results = vector_store.search(
        query_embedding=sample_embeddings[0],
        document_ids=[],
    )

    assert results == []


@pytest.mark.parametrize(
    "document_ids",
    [
        ["doc_001", ""],
        ["doc_001", " "],
        ["doc_001", None],
        ["doc_001", 123],
        "doc_001",
    ],
)
def test_search_rejects_invalid_document_ids(
    vector_store, sample_embeddings, document_ids
):
    """Invalid document ID collections must be rejected."""

    with pytest.raises(ValueError, match="document_ids"):
        vector_store.search(
            query_embedding=sample_embeddings[0],
            document_ids=document_ids,
        )


def test_search_rejects_duplicate_document_ids(
    vector_store, sample_embeddings
):
    """Duplicate document IDs must be rejected."""

    with pytest.raises(ValueError, match="unique"):
        vector_store.search(
            query_embedding=sample_embeddings[0],
            document_ids=["doc_001", "doc_001"],
        )


def test_search_rejects_both_document_filters(
    vector_store, sample_embeddings
):
    """Only one document filter may be provided."""

    with pytest.raises(ValueError, match="either"):
        vector_store.search(
            query_embedding=sample_embeddings[0],
            document_id="doc_001",
            document_ids=["doc_002"],
        )


def test_get_opening_chunks_returns_first_chunks(
    vector_store, sample_chunks, sample_embeddings
):
    """Retrieve the first two chunks in document order."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    results = vector_store.get_opening_chunks(
        document_id="doc_001",
        limit=2,
    )

    assert len(results) == 2

    assert [
        result["metadata"]["chunk_index"]
        for result in results
    ] == [0, 1]

    assert results[0]["text"] == sample_chunks[0].page_content
    assert results[1]["text"] == sample_chunks[1].page_content

    assert all(
        result["metadata"]["document_id"] == "doc_001"
        for result in results
    )


def test_get_opening_chunks_excludes_other_documents(
    vector_store, sample_chunks, sample_embeddings
):
    """Opening chunks must belong only to the requested document."""

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_001",
    )

    vector_store.add_chunks(
        chunks=sample_chunks,
        embeddings=sample_embeddings,
        document_id="doc_002",
    )

    results = vector_store.get_opening_chunks("doc_001")

    assert len(results) == 2

    assert all(
        result["metadata"]["document_id"] == "doc_001"
        for result in results
    )


def test_get_opening_chunks_returns_empty_for_unknown_document(
    vector_store,
):
    results = vector_store.get_opening_chunks("unknown_document")

    assert results == []


@pytest.mark.parametrize(
    "document_id",
    ["", " ", None, 123],
)
def test_get_opening_chunks_rejects_invalid_document_id(
    vector_store, document_id
):
    with pytest.raises(ValueError, match="document_id"):
        vector_store.get_opening_chunks(document_id)


@pytest.mark.parametrize(
    "limit",
    [0, -1, 1.5, True, "2", None],
)
def test_get_opening_chunks_rejects_invalid_limit(
    vector_store, limit
):
    with pytest.raises(ValueError, match="limit"):
        vector_store.get_opening_chunks(
            document_id="doc_001",
            limit=limit,
        )


def test_get_opening_chunks_returns_available_chunks(
    vector_store, sample_chunks, sample_embeddings
):
    """Return fewer chunks when the document contains fewer than limit."""

    vector_store.add_chunks(
        chunks=sample_chunks[:1],
        embeddings=sample_embeddings[:1],
        document_id="doc_001",
    )

    results = vector_store.get_opening_chunks(
        document_id="doc_001",
        limit=5,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["chunk_index"] == 0


