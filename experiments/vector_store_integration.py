
from pathlib import Path

from langchain_core.documents import Document

from backend.app.services.chunking import chunk_documents
from backend.app.services.embedding import embed_documents
from backend.app.services.vector_store import VectorStore
from backend.app.services.embedding import  embed_query


# 1. Create sample source documents
documents = [
    Document(
        page_content=(
            "Machine learning enables computers to learn from data. "
            "Supervised learning uses labeled examples. "
            "Unsupervised learning discovers patterns without labels. "
        ) * 5,
        metadata={"source": "ml_notes.pdf"}
    ),
    Document(
        page_content=(
            "Deep learning uses artificial neural networks. "
            "Neural networks contain interconnected layers. "
            "Training adjusts network weights using optimization. "
        ) * 5,
        metadata={"source": "dl_notes.pdf"}
    )
]

# 2. Generate chunks
chunks = chunk_documents(
    documents,
    chunk_size=100,
    chunk_overlap=20
)

print(f"Generated chunks: {len(chunks)}")

# 3. Generate real BGE embeddings
embeddings = embed_documents(chunks)

print(f"Generated embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")



# 4. Initialize persistent vector store
store = VectorStore(
    persist_directory=Path("experiments/chroma_db")
)

# 5. Store chunks and embeddings
stored_count = store.add_chunks(
    chunks=chunks,
    embeddings=embeddings,
    document_id="integration_test_doc"
)

print(f"Stored chunks: {stored_count}")

# 6. Verify stored records
records = store.collection.get(
    where={"document_id": "integration_test_doc"},
    include=["documents", "metadatas"]
)

print(f"Retrieved records: {len(records['ids'])}")

assert stored_count == len(chunks)
assert len(records["ids"]) == len(chunks)

print("\nIntegration test passed!")

# 7. Generate query embedding
question = "What is supervised learning?"
query_vector = embed_query(question)

# 8. Global semantic search
global_results = store.search(
    query_embedding=query_vector,
    top_k=3
)

print("\nGlobal search results:")

for result in global_results:
    print(f"\nChunk ID: {result['chunk_id']}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Text: {result['text'][:150]}")

# 9. Document-specific search
filtered_results = store.search(
    query_embedding=query_vector,
    top_k=3,
    document_id="integration_test_doc"
)

print(f"\nDocument-specific results: {len(filtered_results)}")

assert len(global_results) <= 3
assert len(filtered_results) <= 3

assert all(
    result["metadata"]["document_id"] == "integration_test_doc"
    for result in filtered_results
)

print("\nSearch integration test passed!")

def test_delete_document(vector_store):
    document_id = "test_delete_doc"

    chunks = [
        Document(
            page_content="Machine learning introduction",
            metadata={"chunk_index": 0}
        ),
        Document(
            page_content="Supervised learning concepts",
            metadata={"chunk_index": 1}
        ),
        Document(
            page_content="Unsupervised learning concepts",
            metadata={"chunk_index": 2}
        ),
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
    ]

    vector_store.add_chunks(chunks, embeddings, document_id)

    deleted_count = vector_store.delete_document(document_id)

    assert deleted_count == 3

    # Repeated deletion should be safe.
    assert vector_store.delete_document(document_id) == 0