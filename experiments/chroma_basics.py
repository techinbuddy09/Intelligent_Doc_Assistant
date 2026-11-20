
import chromadb

# 1. Create a persistent database
client = chromadb.PersistentClient(path="./chroma_db_experiment")

# 2. Create a collection
collection = client.get_or_create_collection(
    name="learning_experiment",
    metadata={"hnsw:space": "cosine"}
)

# 3. Insert sample records
collection.add(
    ids=["chunk_1", "chunk_2", "chunk_3"],
    documents=[
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Python is a popular programming language."
    ],
    embeddings=[
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 0.1, 0.9]
    ],
    metadatas=[
        {"document_id": "doc_1", "chunk_index": 0},
        {"document_id": "doc_1", "chunk_index": 1},
        {"document_id": "doc_2", "chunk_index": 0}
    ]
)

# 4. Perform similarity search
results = collection.query(
    query_embeddings=[[0.85, 0.15, 0.0]],
    n_results=2,
    where={"document_id": "doc_1"}
)

print(results)
