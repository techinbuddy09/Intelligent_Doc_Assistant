from contextlib import asynccontextmanager

from fastapi import FastAPI
from langchain_ollama import ChatOllama

from app.config import settings
from app.api import documents, questions
from app.core.database import initialize_database
from app.services.vector_store import VectorStore
from app.services.document_registry import DocumentRegistry
from app.services.embedding import embed_query
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService

from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources."""

    # 1. Initialize SQLite database.
    initialize_database(settings.SQLITE_DB_PATH)

    # 2. Initialize document registry.
    app.state.document_registry = DocumentRegistry(
        database_path=settings.SQLITE_DB_PATH
    )

    # 3. Initialize ChromaDB.
    app.state.vector_store = VectorStore(
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    # 4. Initialize semantic retrieval.
    app.state.retrieval_service = RetrievalService(
        embedding_function=embed_query,
        vector_store=app.state.vector_store,
        document_registry=app.state.document_registry,
    )

    # 5. Initialize Ollama language model.
    app.state.llm = ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=settings.RAG_TEMPERATURE,
    )

    # 6. Initialize RAG service.
    app.state.rag_service = RAGService(
        retrieval_service=app.state.retrieval_service,
        document_registry=app.state.document_registry,
        llm=app.state.llm,
        top_k=settings.RAG_TOP_K,
    )

    yield

    # Application shutdown.
    app.state.rag_service = None
    app.state.llm = None
    app.state.retrieval_service = None
    app.state.vector_store = None
    app.state.document_registry = None


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    documents.router,
    prefix="/api/v1",
    tags=["Documents"],
)

app.include_router(
    questions.router,
    prefix="/api/v1",
    tags=["Questions"],
)




@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Intelligent Document Assistant is running",
    }

frontend_dist = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "dist"
)

if frontend_dist.exists():
    app.mount(
        "/",
        StaticFiles(directory=frontend_dist, html=True),
        name="frontend",
    )