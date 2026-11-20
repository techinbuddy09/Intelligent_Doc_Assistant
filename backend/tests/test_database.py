
"""
Tests for SQLite database infrastructure.

Tests cover:
    - Database initialization.
    - Table creation.
    - Table schema.
    - Primary key constraints.
    - Unique file hash constraints.
    - Transaction rollback.
    - Connection cleanup.
"""

import sqlite3

import pytest

from app.core.database import (
    initialize_database,
    get_connection,
)


@pytest.fixture
def database_path(tmp_path):
    """Provide an isolated temporary SQLite database."""

    return tmp_path / "test_documents.db"


@pytest.fixture
def initialized_database(database_path):
    """Initialize and provide a temporary database."""

    initialize_database(database_path)

    return database_path


def test_database_creation(database_path):
    """Verify that database initialization creates the file."""

    initialize_database(database_path)

    assert database_path.exists()


def test_documents_table_creation(initialized_database):
    """Verify that the documents table is created."""

    with get_connection(initialized_database) as connection:

        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'documents'
            """
        ).fetchone()

    assert result is not None
    assert result["name"] == "documents"


def test_documents_table_schema(initialized_database):
    """Verify that the documents table has the expected columns."""

    expected_columns = {
        "document_id",
        "filename",
        "file_hash",
        "file_size",
        "status",
        "total_chunks",
        "created_at",
    }

    with get_connection(initialized_database) as connection:

        result = connection.execute(
            "PRAGMA table_info(documents)"
        ).fetchall()

    actual_columns = {
        column["name"]
        for column in result
    }

    assert actual_columns == expected_columns


def test_document_primary_key_constraint(initialized_database):
    """Verify that duplicate document IDs are rejected."""

    with pytest.raises(sqlite3.IntegrityError):

        with get_connection(initialized_database) as connection:

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
                    "doc_001",
                    "first.pdf",
                    "hash_001",
                    1024,
                    "completed",
                ),
            )

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
                    "doc_001",
                    "second.pdf",
                    "hash_002",
                    2048,
                    "completed",
                ),
            )


def test_document_unique_file_hash_constraint(initialized_database):
    """Verify that duplicate file hashes are rejected."""

    with pytest.raises(sqlite3.IntegrityError):

        with get_connection(initialized_database) as connection:

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
                    "doc_001",
                    "first.pdf",
                    "same_hash",
                    1024,
                    "completed",
                ),
            )

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
                    "doc_002",
                    "second.pdf",
                    "same_hash",
                    2048,
                    "completed",
                ),
            )


def test_transaction_rollback(initialized_database):
    """Verify that failed transactions are rolled back."""

    with pytest.raises(RuntimeError, match="Simulated failure"):

        with get_connection(initialized_database) as connection:

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
                    "doc_001",
                    "sample.pdf",
                    "hash_001",
                    1024,
                    "completed",
                ),
            )

            raise RuntimeError("Simulated failure")

    # Verify that the inserted record was rolled back.
    with get_connection(initialized_database) as connection:

        result = connection.execute(
            """
            SELECT *
            FROM documents
            WHERE document_id = ?
            """,
            ("doc_001",),
        ).fetchone()

    assert result is None


def test_connection_cleanup(initialized_database):
    """Verify that database connections close after use."""

    with get_connection(initialized_database) as connection:
        connection.execute("SELECT 1")

    # Accessing a closed connection should raise an error.
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_database_initialization_is_idempotent(database_path):
    """Verify that repeated initialization is safe."""

    initialize_database(database_path)
    initialize_database(database_path)

    with get_connection(database_path) as connection:

        result = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'documents'
            """
        ).fetchall()

    assert len(result) == 1
