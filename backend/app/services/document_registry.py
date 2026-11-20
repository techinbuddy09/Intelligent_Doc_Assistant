
"""
Document metadata registry.

Provides CRUD operations for managing document metadata
using the SQLite database infrastructure.
"""

import sqlite3
from pathlib import Path

from app.core.database import get_connection


class DocumentRegistry:
    """Manage document metadata stored in SQLite."""

    VALID_STATUSES = {"processing", "completed", "failed"}

    def __init__(self, database_path: str | Path):
        """
        Initialize the document registry.

        Args:
            database_path: Path to the SQLite database.
        """
        self.database_path = Path(database_path)

    def register_document(
        self,
        document_id: str,
        filename: str,
        file_hash: str,
        file_size: int,
        status: str = "processing",
    ) -> dict:
        """
        Register a newly uploaded document.

        Returns:
            The registered document record.

        Raises:
            ValueError: If input values are invalid or the
                        document ID/hash already exists.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string.")

        if not isinstance(file_hash, str) or not file_hash.strip():
            raise ValueError("file_hash must be a non-empty string.")

        if (
            not isinstance(file_size, int)
            or isinstance(file_size, bool)
            or file_size < 0
        ):
            raise ValueError("file_size must be a non-negative integer.")

        if status not in self.VALID_STATUSES:
            raise ValueError("Invalid document status.")

        try:
            with get_connection(self.database_path) as connection:
                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id,
                        filename,
                        file_hash,
                        file_size,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        filename,
                        file_hash,
                        file_size,
                        status,
                    ),
                )

                result = connection.execute(
                    """
                    SELECT *
                    FROM documents
                    WHERE document_id = ?
                    """,
                    (document_id,),
                ).fetchone()

                return dict(result)

        except sqlite3.IntegrityError as error:
            raise ValueError(
                "Document ID or file hash already exists."
            ) from error

    def get_document(self, document_id: str) -> dict | None:
        """
        Retrieve a document by its ID.

        Returns:
            Document metadata as a dictionary, or None if absent.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        with get_connection(self.database_path) as connection:
            result = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return dict(result) if result is not None else None

    def get_document_by_hash(self, file_hash: str) -> dict | None:
        """
        Retrieve a document using its unique file hash.

        Returns:
            Document metadata as a dictionary, or None if absent.

        Raises:
            ValueError: If the file hash is invalid.
        """

        if not isinstance(file_hash, str) or not file_hash.strip():
            raise ValueError("file_hash must be a non-empty string.")

        with get_connection(self.database_path) as connection:
            result = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE file_hash = ?
                """,
                (file_hash,),
            ).fetchone()

        return dict(result) if result is not None else None

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """
        Retrieve documents with pagination.

        Args:
            limit: Maximum number of records to return.
            offset: Number of records to skip.

        Returns:
            A list of document metadata dictionaries.
        """

        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
        ):
            raise ValueError("limit must be a positive integer.")

        if (
            not isinstance(offset, int)
            or isinstance(offset, bool)
            or offset < 0
        ):
            raise ValueError("offset must be a non-negative integer.")

        with get_connection(self.database_path) as connection:
            results = connection.execute(
                """
                SELECT *
                FROM documents
                ORDER BY created_at DESC, document_id ASC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in results]

    def update_document_status(
        self,
        document_id: str,
        status: str,
        total_chunks: int | None = None,
    ) -> dict:
        """
        Update a document's ingestion status.

        Args:
            document_id: Document identifier.
            status: New ingestion status.
            total_chunks: Optional number of generated chunks.

        Returns:
            Updated document metadata.

        Raises:
            ValueError: If input is invalid or document doesn't exist.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        if status not in self.VALID_STATUSES:
            raise ValueError("Invalid document status.")

        if total_chunks is not None:
            if (
                not isinstance(total_chunks, int)
                or isinstance(total_chunks, bool)
                or total_chunks < 0
            ):
                raise ValueError(
                    "total_chunks must be a non-negative integer."
                )

        with get_connection(self.database_path) as connection:

            if total_chunks is None:
                cursor = connection.execute(
                    """
                    UPDATE documents
                    SET status = ?
                    WHERE document_id = ?
                    """,
                    (status, document_id),
                )
            else:
                cursor = connection.execute(
                    """
                    UPDATE documents
                    SET status = ?, total_chunks = ?
                    WHERE document_id = ?
                    """,
                    (status, total_chunks, document_id),
                )

            if cursor.rowcount == 0:
                raise ValueError("Document not found.")

            result = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return dict(result)

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document metadata record.

        Returns:
            True if a record was deleted, otherwise False.
        """

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id must be a non-empty string.")

        with get_connection(self.database_path) as connection:
            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            )

        return cursor.rowcount > 0

    def get_completed_document_ids(self) -> list[str]:
        """Return IDs of all successfully ingested documents."""

        with get_connection(self.database_path) as connection:
            results = connection.execute(
                """
                SELECT document_id
                FROM documents
                WHERE status = ?
                ORDER BY document_id ASC
                """,
                ("completed",),
            ).fetchall()

        return [row["document_id"] for row in results]
