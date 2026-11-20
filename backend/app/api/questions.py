import logging

import httpx
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.schemas.question import QuestionRequest, QuestionResponse


router = APIRouter()

logger = logging.getLogger(__name__)


@router.post(
    "/questions",
    response_model=QuestionResponse,
)
async def answer_question(
    payload: QuestionRequest,
    request: Request,
):
    """Answer a question using the uploaded documents."""

    rag_service = request.app.state.rag_service

    try:
        result = await run_in_threadpool(
            rag_service.answer_question,
            question=payload.question,
            document_id=payload.document_id,
        )

        return result

    except ValueError as error:
        if str(error) == "Document not found.":
            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            ) from error

        if str(error) == "Document is not available for retrieval.":
            raise HTTPException(
                status_code=409,
                detail="Document is not available for retrieval.",
            ) from error

        logger.exception("Invalid RAG service input or retrieved data.")

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the question.",
        ) from error

    except (httpx.ConnectError, httpx.TimeoutException) as error:
        logger.exception("Unable to connect to the Ollama service.")

        raise HTTPException(
            status_code=503,
            detail="The language model service is currently unavailable.",
        ) from error

    except Exception as error:
        logger.exception("Unexpected question-answering failure.")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering the question.",
        ) from error