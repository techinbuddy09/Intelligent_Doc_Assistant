
"""
Provides structure-aware document chunking.

Splits Markdown documents using heading boundaries first,
then applies recursive character splitting to oversized
sections while preserving document metadata.
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Split documents into structure-aware chunks.

    Args:
        documents: Extracted LangChain Documents.
        chunk_size: Maximum target chunk length in characters.
        chunk_overlap: Character overlap between chunks
                       within the same section.

    Returns:
        A list of chunked LangChain Documents.

    Raises:
        ValueError: If chunking parameters are invalid.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    # Step 1: Define Markdown heading hierarchy
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    # Step 2: Configure recursive splitting
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    final_chunks = []

    # Step 3: Process each extracted document
    for document_index, document in enumerate(documents):

        if not document.page_content.strip():
            continue

        # Split the document according to Markdown headings
        sections = markdown_splitter.split_text(
            document.page_content
        )

        chunk_index = 0

        # Step 4: Recursively split each section
        for section in sections:

            section_chunks = recursive_splitter.split_documents(
                [section]
            )

            for chunk in section_chunks:

                # Preserve original document metadata
                metadata = {
                    **document.metadata,
                    **chunk.metadata,
                }

                # Store document and chunk identifiers
                metadata["source_document_index"] = document_index
                metadata["chunk_index"] = chunk_index

                # Store the heading hierarchy
                section_headers = [
                    metadata.get(header)
                    for header in [
                        "Header 1",
                        "Header 2",
                        "Header 3",
                        "Header 4",
                    ]
                    if metadata.get(header)
                ]

                metadata["section"] = (
                    " > ".join(section_headers)
                    if section_headers
                    else "General"
                )

                final_chunks.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=metadata,
                    )
                )

                chunk_index += 1

    return final_chunks
