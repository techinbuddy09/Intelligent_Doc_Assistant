
"""
SQLite database infrastructure.

Handles database initialization, schema creation,
and managed database connections.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator


def initialize_database(database_path: str | Path) -> None:
    """
    Initialize the SQLite database and create required tables.

    Args:
        database_path: Location of the SQLite database file.
    """

    database_path = Path(database_path)

    # Ensure the parent directory exists.
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(database_path) as connection:

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL CHECK (file_size >= 0),
                status TEXT NOT NULL,
                total_chunks INTEGER NOT NULL DEFAULT 0
                    CHECK (total_chunks >= 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Improve document listing by upload timestamp.
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_created_at
            ON documents(created_at)
            """
        )


@contextmanager
def get_connection(
    database_path: str | Path,
) -> Iterator[sqlite3.Connection]:
    """
    Provide a managed SQLite database connection.

    Automatically commits successful transactions,
    rolls back failed transactions, and closes the connection.
    """

    connection = sqlite3.connect(
        str(database_path),
        timeout=30,
    )

    connection.row_factory = sqlite3.Row

    # Enable foreign key enforcement.
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()
