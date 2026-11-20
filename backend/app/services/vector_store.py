
"""
ChromaDB vector storage service.

Responsibilities:
    - Persistent vector storage.
    - Document chunk insertion and updates.
    - Global semantic search.
    - Single-document semantic search.
    - Multiple-document semantic search.
    - Embedding dimension validation.
    - Document deletion.
"""

from importlib.metadata import metadata
import math
from collections.abc import Sequence
from pathlib import Path

import chromadb
from langchain_core.documents import Document


class VectorStore:
    """Manage document embeddings using persistent ChromaDB storage."""

    def __init__(self, persist_directory: str | Path):
        """
        Initialize a persistent ChromaDB vector store.
        """

        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name="document_chunks",
            metadata={"hnsw:space": "cosine"},
        )

        self._embedding_dimension = (
            self._get_collection_embedding_dimension()
        )

    def _get_collection_embedding_dimension(self) -> int | None:
        """
        Retrieve the dimensionality of an existing collection.

        Returns None when the collection contains no embeddings.
        """

        result = self.collection.get(
            limit=1,
            include=["embeddings"],
        )

        embeddings = result["embeddings"]

        if embeddings is None or len(embeddings) == 0:
            return None

        return len(embeddings[0])

    def _validate_embedding_dimensions(
        self,
        embeddings: Sequence[Sequence[float]],
        expected_dimension: int | None = None,
    ) -> int | None:
        """
        Validate embedding dimensionality.

        Ensures all supplied embeddings have identical dimensions
        and optionally match the expected collection dimension.

        Returns the validated dimension.
        """

        if not embeddings:
            return None

        dimensions = {
            len(embedding)
            for embedding in embeddings
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All embeddings must have the same dimensions."
            )

        actual_dimension = dimensions.pop()

        if (
            expected_dimension is not None
            and actual_dimension != expected_dimension
        ):
            raise ValueError(
                f"Embedding dimension mismatch: expected "
                f"{expected_dimension}, received {actual_dimension}."
            )

        return actual_dimension

    def add_chunks(
        self,
        chunks: Sequence[Document],
        embeddings: Sequence[Sequence[float]],
        document_id: str,
    ) -> int:
        """
        Store document chunks and their corresponding embeddings.

        Chunk IDs include the document ID, source document index,
        and chunk index to prevent collisions when multiple source
        documents are processed together.

        Upsert makes repeated ingestion of the same chunk IDs
        idempotent at the record level.

        Returns:
            Number of inserted or updated chunks.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "document_id must be a non-empty string."
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks must match number of embeddings."
            )

        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        vectors = []

        for chunk, embedding in zip(chunks, embeddings):

            if not chunk.page_content.strip():
                raise ValueError(
                    "Chunk content cannot be empty."
                )

            metadata = dict(chunk.metadata)

            chunk_index = metadata.get("chunk_index")
            source_index = metadata.get("source_document_index", 0)
            metadata["source_document_index"] = source_index

            if (
                not isinstance(chunk_index, int)
                or isinstance(chunk_index, bool)
                or chunk_index < 0
            ):
                raise ValueError(
                    "Every chunk must have a valid chunk_index."
                )

            if (
                not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or source_index < 0
            ):
                raise ValueError(
                    "source_document_index must be a non-negative integer."
                )

            # Validate ChromaDB metadata values.
            for key, value in metadata.items():

                if not isinstance(key, str) or not isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    raise ValueError(
                        f"Unsupported metadata value for key: {key}"
                    )

                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(
                        f"Metadata value for {key} must be finite."
                    )

            # Generate deterministic chunk IDs.
            chunk_id = (
                f"{document_id}_source_{source_index}_chunk_{chunk_index}"
            )

            ids.append(chunk_id)
            documents.append(chunk.page_content)

            # Associate the chunk with its parent document.
            metadata["document_id"] = document_id
            metadatas.append(metadata)

            # Convert NumPy arrays or array-like vectors into lists.
            vector = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )

            vector = [float(value) for value in vector]

            if not vector:
                raise ValueError(
                    "Embedding cannot be empty."
                )

            if not all(math.isfinite(value) for value in vector):
                raise ValueError(
                    "Embedding contains non-finite values."
                )

            vectors.append(vector)

        # Ensure generated chunk IDs are unique.
        if len(set(ids)) != len(ids):
            raise ValueError(
                "Duplicate chunk IDs detected."
            )

        # Validate embedding dimensions.
        embedding_dimension = self._validate_embedding_dimensions(
            embeddings=vectors,
            expected_dimension=self._embedding_dimension,
        )

        # Insert new records or update existing records.
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=vectors,
            metadatas=metadatas,
        )

        # Establish the collection dimension after first insertion.
        if self._embedding_dimension is None:
            self._embedding_dimension = embedding_dimension

        return len(ids)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        document_id: str | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks using vector similarity.

        Supports three retrieval modes:

            1. Global search:
               Search across the entire collection.

            2. Single-document search:
               Restrict retrieval to one document.

            3. Multiple-document search:
               Restrict retrieval to a specified collection of documents.

        Returns:
            A list containing chunk IDs, original text,
            metadata, and cosine distances.

            Lower cosine distance indicates greater similarity.
        """

        # 1. Validate top_k.
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        # 2. Validate mutually exclusive document filters.
        if document_id is not None and document_ids is not None:
            raise ValueError(
                "Provide either document_id or document_ids, not both."
            )

        # 3. Build optional metadata filter.
        query_filter = None

        if document_id is not None:

            if (
                not isinstance(document_id, str)
                or not document_id.strip()
            ):
                raise ValueError(
                    "document_id cannot be empty."
                )

            query_filter = {
                "document_id": document_id
            }

        elif document_ids is not None:

            if (
                not isinstance(document_ids, Sequence)
                or isinstance(document_ids, (str, bytes))
            ):
                raise ValueError(
                    "document_ids must be a sequence of strings."
                )

            if not document_ids:
                return []

            if any(
                not isinstance(doc_id, str) or not doc_id.strip()
                for doc_id in document_ids
            ):
                raise ValueError(
                    "document_ids must contain non-empty strings."
                )

            if len(set(document_ids)) != len(document_ids):
                raise ValueError(
                    "document_ids must contain unique values."
                )

            query_filter = {
                "document_id": {
                    "$in": list(document_ids)
                }
            }

        # 4. Convert query embedding into a Python list.
        vector = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else list(query_embedding)
        )

        vector = [float(value) for value in vector]

        if not vector:
            raise ValueError(
                "Query embedding cannot be empty."
            )

        if not all(math.isfinite(value) for value in vector):
            raise ValueError(
                "Query embedding contains non-finite values."
            )

        # 5. Validate query embedding dimensions.
        self._validate_embedding_dimensions(
            embeddings=[vector],
            expected_dimension=self._embedding_dimension,
        )

        # 6. Execute similarity search.
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=query_filter,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        # 7. Convert ChromaDB's nested response into a clean list.
        retrieved_chunks = []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "distance": float(distance),
                }
            )

        return retrieved_chunks

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks associated with a document.

        Args:
            document_id: Unique identifier of the document.

        Returns:
            Number of deleted chunks.

        Raises:
            ValueError: If document_id is invalid.
        """

        # 1. Validate document ID.
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "document_id must be a non-empty string."
            )

        # 2. Retrieve all chunk IDs belonging to this document.
        result = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        chunk_ids = result["ids"]

        # 3. Handle nonexistent documents.
        if not chunk_ids:
            return 0

        # 4. Delete matching chunks.
        self.collection.delete(
            ids=chunk_ids
        )

        # 5. Return deletion count.
        return len(chunk_ids)

    def get_opening_chunks(
        self,
        document_id: str,
        limit: int = 2,
    ) -> list[dict]:
        """
        Retrieve the opening chunks of a document.

        Uses ChromaDB metadata filtering instead of generating
        additional embeddings or performing another similarity search.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
        ):
            raise ValueError("limit must be a positive integer.")

        results = self.collection.get(
            where={
                "$and": [
                    {"document_id": {"$eq": document_id}},
                    {"source_document_index": {"$eq": 0}},
                    {"chunk_index": {"$lte": limit - 1}},
                ]
            },
            include=["documents", "metadatas"],
        )

        chunks = []

        for chunk_id, text, metadata in zip(
            results["ids"],
            results["documents"],
            results["metadatas"],
        ):
            chunks.append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": None,
            })

        chunks.sort(
            key=lambda chunk: chunk["metadata"]["chunk_index"]
        )

        return chunks

    
