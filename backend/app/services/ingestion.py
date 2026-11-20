
"""
Document ingestion orchestration service.

Coordinates document loading, chunking, embedding generation,
and vector storage into a single ingestion workflow.
"""

from pathlib import Path

from app.services.document_loader import load_pdf_documents
from app.services.chunking import chunk_documents
from app.services.embedding import embed_documents
from app.services.vector_store import VectorStore


def ingest_pdf(
    pdf_path: str | Path,
    document_id: str,
    vector_store: VectorStore,
) -> dict:
    """
    Load, process, embed, and store a PDF document.

    Args:
        pdf_path: Path to the PDF file.
        document_id: Unique identifier for the document.
        vector_store: Initialized ChromaDB vector store.

    Returns:
        A summary of the ingestion operation.

    Raises:
        ValueError: If document_id is invalid or no chunks are generated.
        FileNotFoundError: If the PDF does not exist.
        DocumentLoadingError: If document loading fails.
    """

    # Step 1: Validate document ID.
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    # Step 2: Load PDF documents.
    documents = load_pdf_documents(pdf_path)

    # Step 3: Split documents into chunks.
    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were generated from the PDF.")

    # Step 4: Generate embeddings for the chunks.
    embeddings = embed_documents(chunks)

    # Step 5: Store chunks and embeddings in ChromaDB.
    stored_chunks = vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
    )

    # Step 6: Return ingestion summary.
    return {
        "document_id": document_id,
        "documents_loaded": len(documents),
        "total_chunks": len(chunks),
        "total_embeddings": len(embeddings),
        "stored_chunks": stored_chunks,
        "status": "completed",
    }
