
"""
PDF document extraction using Docling.

Converts PDF files into Markdown-backed LangChain Documents
while preserving document structure and metadata.
"""

from pathlib import Path

import torch

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.documents import Document


class DocumentLoadingError(Exception):
    """Raised when a PDF cannot be extracted."""


def _build_pdf_pipeline_options() -> PdfPipelineOptions:
    """
    Configure Docling's PDF extraction pipeline.

    Uses GPU acceleration when available and configures
    table extraction for complex PDF layouts.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(
            device=device,
            num_threads=4,
        ),
        do_table_structure=True,
        do_formula_enrichment=False,
        do_ocr=False,
    )

    # Let TableFormer determine table cell content instead of
    # matching its predictions against potentially unreliable
    # PDF text-cell coordinates.
    pipeline_options.table_structure_options.do_cell_matching = False

    # Prefer extraction accuracy over table processing speed.
    pipeline_options.table_structure_options.mode = (
        TableFormerMode.ACCURATE
    )

    return pipeline_options


def load_pdf_documents(pdf_path: str | Path) -> list[Document]:
    """
    Extract a PDF and return its content as LangChain Documents.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of LangChain Document objects containing
        extracted Markdown and metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        DocumentLoadingError: If extraction fails or produces
            no usable documents.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise DocumentLoadingError(
            "Provided path is not a file."
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise DocumentLoadingError(
            "Only PDF files are supported."
        )

    try:
        pipeline_options = _build_pdf_pipeline_options()

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        loader = DoclingLoader(
            file_path=str(pdf_path),
            converter=converter,
            export_type=ExportType.MARKDOWN,
        )

        documents = loader.load()

        if not documents:
            raise DocumentLoadingError(
                "No documents were extracted from the PDF."
            )

        usable_documents = [
            document
            for document in documents
            if document.page_content
            and document.page_content.strip()
        ]

        if not usable_documents:
            raise DocumentLoadingError(
                "PDF extraction produced no usable text."
            )

        return usable_documents

    except DocumentLoadingError:
        raise

    except Exception as error:
        raise DocumentLoadingError(
            f"Unable to load PDF: {error}"
        ) from error
