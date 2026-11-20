
"""
Inspects the underlying DoclingDocument representation.

Investigates document elements, hierarchy,
page information, tables and pictures.
"""

from pathlib import Path

import pytest

from docling.document_converter import DocumentConverter


@pytest.mark.integration
def test_inspect_docling_document():

    pdf_path = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"

    # 1. Initialize Docling's converter
    converter = DocumentConverter()

    # 2. Convert PDF into Docling's native representation
    result = converter.convert(source=str(pdf_path))

    doc = result.document

    assert doc is not None

    print("\n" + "=" * 60)
    print("DOCLING DOCUMENT INSPECTION")
    print("=" * 60)

    # 3. Basic document information
    print("\n[DOCUMENT INFORMATION]")

    print("Document name:", doc.name)
    print("Number of pages:", len(doc.pages))
    print("Page numbers:", list(doc.pages.keys()))

    # 4. Inspect available collections
    print("\n[DOCUMENT ELEMENT COUNTS]")

    print("Text elements:", len(doc.texts))
    print("Tables:", len(doc.tables))
    print("Pictures:", len(doc.pictures))
    print("Groups:", len(doc.groups))

    # 5. Inspect document hierarchy
    print("\n[DOCUMENT HIERARCHY]")

    for item, level in doc.iterate_items():

        label = getattr(item, "label", None)
        text = getattr(item, "text", None)

        print("\n" + "  " * level + f"Label: {label}")

        if text:
            print(
                "  " * level
                + f"Text: {str(text)[:200]}"
            )

        # 6. Inspect page provenance
        provenance = getattr(item, "prov", [])

        if provenance:

            for prov in provenance:

                print(
                    "  " * level
                    + f"Page: {prov.page_no}"
                )

                print(
                    "  " * level
                    + f"Bounding box: {prov.bbox}"
                )

    # 7. Basic validation
    assert len(doc.pages) > 0

    print("\n" + "=" * 60)
    print("DOCLING STRUCTURE INSPECTION COMPLETED")
    print("=" * 60)
