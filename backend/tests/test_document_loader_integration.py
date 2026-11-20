
"""
Integration tests for the document loader.

Inspects the actual LangChain Document output
produced by the Docling integration.
"""

from pathlib import Path
from pprint import pprint

import pytest

from langchain_core.documents import Document
from app.services.document_loader import load_pdf_documents


@pytest.mark.integration
def test_load_real_pdf():

    pdf_path = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"

    documents = load_pdf_documents(pdf_path)

    # 1. Basic validation
    assert len(documents) > 0

    assert all(
        isinstance(document, Document)
        for document in documents
    )

    print("\n" + "=" * 60)
    print("DOCUMENT INSPECTION")
    print("=" * 60)

    print(f"\nTotal LangChain Documents: {len(documents)}")

    # 2. Inspect each LangChain Document
    for index, document in enumerate(documents, start=1):

        print("\n" + "-" * 60)
        print(f"DOCUMENT {index}")
        print("-" * 60)

        # 3. Inspect page_content
        print("\n[PAGE CONTENT]")

        print(f"Content type: {type(document.page_content)}")
        print(f"Content length: {len(document.page_content)} characters")

        print("\nExtracted Markdown:")
        print(document.page_content)

        # 4. Inspect metadata
        print("\n[METADATA]")

        print(f"Metadata type: {type(document.metadata)}")

        print("\nMetadata keys:")
        print(list(document.metadata.keys()))

        print("\nComplete metadata:")
        pprint(document.metadata, sort_dicts=False)

        # 5. Basic validation
        assert document.page_content.strip()
        assert isinstance(document.metadata, dict)

    print("\n" + "=" * 60)
    print("DOCUMENT INSPECTION COMPLETED")
    print("=" * 60)
