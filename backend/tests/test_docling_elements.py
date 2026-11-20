
"""
Integration experiments for Docling table and picture extraction.

Experiments:
1. Extract structured table data.
2. Export tables as Markdown and CSV.
3. Extract picture images.
4. Preserve page provenance and bounding boxes.

The PDF is converted only once per test module.
"""

from pathlib import Path

import pytest

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from docling_core.types.doc import ImageRefMode


PDF_PATH = Path(__file__).resolve().parent / "fixtures" / "sample.pdf"
OUTPUT_DIR = Path("tests/outputs/docling_elements")


@pytest.fixture(scope="module")
def docling_document():
    """
    Convert the sample PDF once and share the DoclingDocument
    between the experiments.
    """

    assert PDF_PATH.exists(), f"PDF not found: {PDF_PATH}"

    pipeline_options = PdfPipelineOptions()

    # Required for extracting images from PDF pages.
    pipeline_options.generate_page_images = True

    # Required for extracting picture elements.
    pipeline_options.generate_picture_images = True

    # Higher resolution for image inspection.
    pipeline_options.images_scale = 2.0

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(PDF_PATH)

    assert result.document is not None

    return result.document


def get_provenance(element):
    """
    Extract page number and bounding box from an element.
    """

    if not element.prov:
        return {
            "page": None,
            "bbox": None,
        }

    provenance = element.prov[0]

    return {
        "page": provenance.page_no,
        "bbox": provenance.bbox,
    }


@pytest.mark.integration
def test_extract_tables(docling_document, tmp_path):
    """
    Experiment A:
    Inspect structured table contents and export them.
    """

    doc = docling_document

    tables = doc.tables

    print("\n========== TABLE EXPERIMENT ==========")
    print(f"Tables detected: {len(tables)}")

    assert len(tables) > 0, "No tables detected in the PDF."

    table_output_dir = tmp_path / "tables"
    table_output_dir.mkdir(parents=True, exist_ok=True)

    for index, table in enumerate(tables, start=1):

        print(f"\n---------- TABLE {index} ----------")

        # Page number and bounding box
        provenance = get_provenance(table)

        print("Page:", provenance["page"])
        print("Bounding box:", provenance["bbox"])

        # Caption
        caption = table.caption_text(doc)

        print("Caption:", caption)

        # Structured table extraction
        dataframe = table.export_to_dataframe(doc=doc)

        print("\nDataFrame shape:", dataframe.shape)
        print("\nExtracted table:")
        print(dataframe.to_string(index=False))

        # Save structured data
        csv_path = table_output_dir / f"table_{index}.csv"

        dataframe.to_csv(csv_path, index=False)

        print("CSV saved:", csv_path)

        # Markdown export
        markdown = table.export_to_markdown(doc=doc)

        print("\nMarkdown representation:")
        print(markdown)

        markdown_path = (
            table_output_dir / f"table_{index}.md"
        )

        markdown_path.write_text(
            markdown,
            encoding="utf-8"
        )

        # Extract table image, if available
        table_image = table.get_image(doc)

        if table_image is not None:

            image_path = (
                table_output_dir / f"table_{index}.png"
            )

            table_image.save(image_path)

            print("Table image saved:", image_path)

        else:
            print("Table image unavailable.")

    print("\nTABLE EXPERIMENT COMPLETED")


@pytest.mark.integration
def test_extract_pictures(docling_document, tmp_path):
    """
    Experiment B:
    Inspect picture elements and extract their images.
    """

    doc = docling_document

    pictures = doc.pictures

    print("\n========== PICTURE EXPERIMENT ==========")
    print(f"Pictures detected: {len(pictures)}")

    assert len(pictures) > 0, "No pictures detected in the PDF."

    picture_output_dir = tmp_path / "pictures"
    picture_output_dir.mkdir(parents=True, exist_ok=True)

    extracted_count = 0

    for index, picture in enumerate(pictures, start=1):

        print(f"\n---------- PICTURE {index} ----------")

        # Page number and bounding box
        provenance = get_provenance(picture)

        print("Page:", provenance["page"])
        print("Bounding box:", provenance["bbox"])

        # Caption
        caption = picture.caption_text(doc)

        print("Caption:", caption)

        # Extract actual picture image
        image = picture.get_image(doc)

        if image is None:

            print("Image extraction returned None.")
            continue

        image_path = (
            picture_output_dir / f"picture_{index}.png"
        )

        image.save(image_path)

        extracted_count += 1

        print("Image saved:", image_path)
        print("Image dimensions:", image.size)

    print("\nPictures detected:", len(pictures))
    print("Pictures extracted:", extracted_count)

    assert extracted_count > 0, (
        "Picture elements were detected, but no images "
        "could be extracted. Check the pipeline options."
    )

    print("\nPICTURE EXPERIMENT COMPLETED")
