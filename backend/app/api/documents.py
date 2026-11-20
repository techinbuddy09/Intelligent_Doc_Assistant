
"""
Document management API.

Provides:
    - PDF upload and validation.
    - Duplicate document detection.
    - Document ingestion.
    - Document listing with pagination.
    - Document deletion across SQLite, ChromaDB, and filesystem.
"""

import hashlib
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.services.document_loader import DocumentLoadingError
from app.services.ingestion import ingest_pdf


router = APIRouter()
logger = logging.getLogger(__name__)

UPLOAD_DIR = settings.UPLOAD_DIR
MAX_FILE_SIZE = settings.MAX_UPLOAD_SIZE


def calculate_file_hash(contents: bytes) -> str:
    """Generate a SHA-256 hash of file contents."""
    return hashlib.sha256(contents).hexdigest()


# ---------------------------------------------------------------------
# Upload document
# ---------------------------------------------------------------------

@router.post("/documents/upload", status_code=201)
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    """Upload, validate, and ingest a PDF document."""

    file_path = None
    document_id = None
    registry = None
    vector_store = None
    is_retry = False

    try:
        # 1. Validate file extension.
        if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported.",
            )

        # 2. Read uploaded file with a size limit.
        contents = await file.read(MAX_FILE_SIZE + 1)

        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the {MAX_FILE_SIZE} byte limit.",
            )

        # 3. Validate PDF signature.
        if not contents.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF file.",
            )

        # 4. Prepare storage.
        await run_in_threadpool(
            UPLOAD_DIR.mkdir,
            parents=True,
            exist_ok=True,
        )

        # 5. Calculate file hash.
        file_hash = calculate_file_hash(contents)

        # 6. Retrieve application services.
        registry = request.app.state.document_registry
        vector_store = request.app.state.vector_store

        # 7. Check for an existing document.
        existing_document = await run_in_threadpool(
            registry.get_document_by_hash,
            file_hash,
        )

        if existing_document is not None:
            existing_status = existing_document["status"]
            document_id = existing_document["document_id"]

            if existing_status == "completed":
                response.status_code = 200

                return {
                    "document_id": document_id,
                    "filename": file.filename,
                    "file_size": len(contents),
                    "status": "duplicate",
                    "message": "This document has already been uploaded.",
                }

            if existing_status == "processing":
                raise HTTPException(
                    status_code=409,
                    detail="This document is already being processed.",
                )

            if existing_status == "failed":
                is_retry = True

                await run_in_threadpool(
                    registry.update_document_status,
                    document_id,
                    "processing",
                )

                logger.info(
                    "Retrying failed document: %s",
                    document_id,
                )

            else:
                raise HTTPException(
                    status_code=409,
                    detail="Document is in an unsupported state.",
                )

        else:
            # 8. Register a new document.
            document_id = str(uuid4())

            await run_in_threadpool(
                registry.register_document,
                document_id=document_id,
                filename=file.filename,
                file_hash=file_hash,
                file_size=len(contents),
                status="processing",
            )

        # 9. Save the uploaded PDF.
        file_path = UPLOAD_DIR / f"{document_id}.pdf"

        await run_in_threadpool(
            file_path.write_bytes,
            contents,
        )

        # 10. Remove old vectors before retrying ingestion.
        if is_retry:
            await run_in_threadpool(
                vector_store.delete_document,
                document_id,
            )

        logger.info("Document uploaded: %s", document_id)

        # 11. Execute ingestion.
        ingestion_result = await run_in_threadpool(
            ingest_pdf,
            file_path,
            document_id,
            vector_store,
        )

        # 12. Mark document as completed.
        await run_in_threadpool(
            registry.update_document_status,
            document_id=document_id,
            status="completed",
            total_chunks=ingestion_result["total_chunks"],
        )

        logger.info(
            "Document ingestion completed: %s | Chunks: %d | Stored: %d",
            document_id,
            ingestion_result["total_chunks"],
            ingestion_result["stored_chunks"],
        )

        # 13. Return ingestion summary.
        return {
            "document_id": document_id,
            "filename": file.filename,
            "file_size": len(contents),
            **ingestion_result,
            "message": "Document uploaded and ingested successfully.",
        }

    except HTTPException:
        raise

    except DocumentLoadingError as error:
        logger.exception(
            "Document loading failed: %s",
            document_id,
        )

        if registry is not None and document_id is not None:
            await run_in_threadpool(
                registry.update_document_status,
                document_id,
                "failed",
            )

        if file_path is not None:
            await run_in_threadpool(
                file_path.unlink,
                missing_ok=True,
            )

        raise HTTPException(
            status_code=422,
            detail="Document processing failed.",
        ) from error

    except ValueError as error:
        logger.exception(
            "Document ingestion validation failed: %s",
            document_id,
        )

        if registry is not None and document_id is not None:
            await run_in_threadpool(
                registry.update_document_status,
                document_id,
                "failed",
            )

        if file_path is not None:
            await run_in_threadpool(
                file_path.unlink,
                missing_ok=True,
            )

        raise HTTPException(
            status_code=422,
            detail="Document ingestion failed.",
        ) from error

    except Exception as error:
        logger.exception(
            "Unexpected document upload or ingestion failure: %s",
            document_id,
        )

        if registry is not None and document_id is not None:
            await run_in_threadpool(
                registry.update_document_status,
                document_id,
                "failed",
            )

        if file_path is not None:
            await run_in_threadpool(
                file_path.unlink,
                missing_ok=True,
            )

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during document ingestion.",
        ) from error

    finally:
        await file.close()


# ---------------------------------------------------------------------
# List documents
# ---------------------------------------------------------------------

@router.get("/documents")
async def list_documents(
    request: Request,
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List registered documents with pagination."""

    registry = request.app.state.document_registry

    documents = await run_in_threadpool(
        registry.list_documents,
        limit,
        offset,
    )

    return {
        "documents": [
            {
                "document_id": document["document_id"],
                "filename": document["filename"],
                "file_size": document["file_size"],
                "status": document["status"],
                "total_chunks": document["total_chunks"],
                "created_at": document["created_at"],
            }
            for document in documents
        ],
        "limit": limit,
        "offset": offset,
    }


# ---------------------------------------------------------------------
# Delete document
# ---------------------------------------------------------------------

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
):
    """Delete a document and its associated data."""

    registry = request.app.state.document_registry

    # 1. Retrieve document metadata.
    document = await run_in_threadpool(
        registry.get_document,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    # 2. Prevent deletion during ingestion.
    if document["status"] == "processing":
        raise HTTPException(
            status_code=409,
            detail="Cannot delete a document while it is processing.",
        )

    # 3. Access the vector store only after validation.
    vector_store = request.app.state.vector_store

    # 4. Construct and validate the stored file path.
    file_path = (UPLOAD_DIR / f"{document_id}.pdf").resolve()
    upload_directory = UPLOAD_DIR.resolve()

    if file_path.parent != upload_directory:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        )

    try:
        # 5. Delete associated ChromaDB vectors.
        deleted_chunks = await run_in_threadpool(
            vector_store.delete_document,
            document_id,
        )

        # 6. Delete the uploaded PDF.
        await run_in_threadpool(
            file_path.unlink,
            missing_ok=True,
        )

        # 7. Delete SQLite metadata.
        deleted = await run_in_threadpool(
            registry.delete_document,
            document_id,
        )

        if not deleted:
            raise RuntimeError(
                "Document metadata could not be deleted."
            )

        logger.info(
            "Document deleted: %s | Chunks removed: %d",
            document_id,
            deleted_chunks,
        )

        return {
            "document_id": document_id,
            "deleted_chunks": deleted_chunks,
            "message": "Document deleted successfully.",
        }

    except Exception as error:
        logger.exception(
            "Document deletion failed: %s",
            document_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document deletion failed.",
        ) from error
