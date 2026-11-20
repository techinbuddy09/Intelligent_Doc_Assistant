
from collections.abc import Callable

from app.services.document_registry import DocumentRegistry
from app.services.vector_store import VectorStore


class RetrievalService:
    """Coordinate query embedding and document retrieval."""

    OVERVIEW_QUERY_PATTERNS = (
        "overview",
        "summarize",
        "summary",
        "main subject",
        "main topic",
        "main areas",
        "key areas",
        "main topics",
        "key topics",
        "topics covered",
        "areas covered",
        "what does this document cover",
        "what is this document about",
        "describe this document",
        "contents of this document",
    )

    def __init__(
        self,
        embedding_function: Callable[[str], list[float]],
        vector_store: VectorStore,
        document_registry: DocumentRegistry,
    ):
        self.embedding_function = embedding_function
        self.vector_store = vector_store
        self.document_registry = document_registry

    @staticmethod
    def _is_overview_query(query: str) -> bool:
        """Identify broad document-level questions using simple rules."""
        normalized_query = " ".join(query.lower().split())

        return any(
            pattern in normalized_query
            for pattern in RetrievalService.OVERVIEW_QUERY_PATTERNS
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """Retrieve relevant chunks from completed documents."""

        # 1. Validate query.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        # 2. Validate top_k.
        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer.")

        # 3. Handle document-specific retrieval.
        if document_id is not None:
            if not isinstance(document_id, str) or not document_id.strip():
                raise ValueError(
                    "document_id must be a non-empty string."
                )

            document = self.document_registry.get_document(document_id)

            if document is None:
                raise ValueError("Document not found.")

            if document["status"] != "completed":
                raise ValueError(
                    "Document is not available for retrieval."
                )

            # Broad questions use the document's opening context.
            # This avoids unnecessary embedding and semantic search.
            if self._is_overview_query(query):
                return self.vector_store.get_opening_chunks(
                    document_id=document_id,
                    limit=min(top_k, 2),
                )

            document_ids = None

        # 4. Handle global retrieval.
        else:
            document_ids = (
                self.document_registry.get_completed_document_ids()
            )

            if not document_ids:
                return []

        # 5. Generate query embedding for semantic retrieval.
        query_embedding = self.embedding_function(query)

        # 6. Execute filtered vector search.
        if document_id is not None:
            return self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                document_id=document_id,
            )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
        )
