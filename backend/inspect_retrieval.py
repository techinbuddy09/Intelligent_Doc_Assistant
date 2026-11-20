
from app.config import settings
from app.services.embedding import embed_query
from app.services.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.retrieval_service import RetrievalService


# Initialize existing persistent services.
vector_store = VectorStore(
    settings.CHROMA_PERSIST_DIR
)

document_registry = DocumentRegistry(
    settings.SQLITE_DB_PATH
)

retrieval_service = RetrievalService(
    embedding_function=embed_query,
    vector_store=vector_store,
    document_registry=document_registry,
)


# Initial retrieval evaluation queries.
queries = [
    "What is a hypothesis space?",
    "What is empirical risk minimization?",
    "Explain cross-validation.",
    "What is the difference between frequentist and Bayesian probability?",
]


for query in queries:
    print("\n" + "=" * 100)
    print(f"QUERY: {query}")
    print("=" * 100)

    results = retrieval_service.retrieve(
        query=query,
        top_k=5,
    )

    if not results:
        print("No results found.")
        continue

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"\nRANK: {rank}")
        print(f"CHUNK ID: {result['chunk_id']}")
        print(f"DISTANCE: {result['distance']:.4f}")
        print(f"SECTION: {metadata.get('section', 'Unknown')}")
        print("-" * 80)
        print(result["text"])
