from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    """Request body for document question answering."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question to answer using uploaded documents.",
    )

    document_id: str | None = Field(
        default=None,
        description="Optional document ID to restrict the search.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Question cannot be empty.")

        return value

    @field_validator("document_id")
    @classmethod
    def validate_document_id(cls, value: str | None) -> str | None:
        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError("document_id cannot be empty.")

        return value


class SourceReference(BaseModel):
    """Document source used to generate an answer."""

    document_id: str
    filename: str | None = None
    chunk_ids: list[str]


class QuestionResponse(BaseModel):
    """Response returned by the question-answering endpoint."""

    question: str
    answer: str
    sources: list[SourceReference]