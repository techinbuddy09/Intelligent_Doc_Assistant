
from pathlib import Path

from pydantic_settings import BaseSettings


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):

    APP_NAME: str = "Intelligent Document Assistant"

    APP_VERSION: str = "1.0.0"

    DEBUG: bool = True

    # File storage
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"

    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50 MB

    # Vector database storage
    CHROMA_PERSIST_DIR: Path = BASE_DIR / "data" / "chroma_db"

    # SQLite database storage
    SQLITE_DB_PATH: Path = BASE_DIR / "data" / "documents.db"

    # LLM / RAG configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    OLLAMA_MODEL: str = "gemma3:4b"

    RAG_TOP_K: int = 5

    RAG_TEMPERATURE: float = 0.0


settings = Settings()
