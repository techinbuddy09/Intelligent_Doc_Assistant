
"""
Tests document loading functionality.

Verifies successful PDF loading, extracted content,
metadata, and invalid file handling.
"""

import pytest

from langchain_core.documents import Document

from app.services import document_loader


def test_load_pdf_documents(tmp_path, monkeypatch):

    # Create a temporary PDF path
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-")

    # Expected loader output
    expected_documents = [
        Document(
            page_content="Introduction to Machine Learning",
            metadata={"source": str(pdf_path)}
        )
    ]

    # Mock DoclingLoader
    class MockLoader:

        def __init__(self, file_path, converter, export_type):
            self.file_path = file_path
            self.converter = converter
            self.export_type = export_type

        def load(self):
            return expected_documents

    monkeypatch.setattr(
        document_loader,
        "DoclingLoader",
        MockLoader
    )

    # Execute
    documents = document_loader.load_pdf_documents(pdf_path)

    # Verify
    assert len(documents) == 1

    assert isinstance(documents[0], Document)

    assert documents[0].page_content == (
        "Introduction to Machine Learning"
    )

    assert documents[0].metadata["source"] == str(pdf_path)


def test_pdf_not_found():

    with pytest.raises(FileNotFoundError):

        document_loader.load_pdf_documents(
            "nonexistent.pdf"
        )


def test_invalid_file_extension(tmp_path):

    file_path = tmp_path / "sample.txt"
    file_path.write_text("Sample content")

    with pytest.raises(document_loader.DocumentLoadingError):

        document_loader.load_pdf_documents(file_path)


def test_empty_loader_result(tmp_path, monkeypatch):

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-")

    class MockLoader:

        def __init__(self, file_path, converter, export_type):
            pass

        def load(self):
            return []

    monkeypatch.setattr(
        document_loader,
        "DoclingLoader",
        MockLoader
    )

    with pytest.raises(document_loader.DocumentLoadingError):

        document_loader.load_pdf_documents(pdf_path)


def test_loader_failure(tmp_path, monkeypatch):

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-")

    class MockLoader:

        def __init__(self, file_path, converter, export_type):
            pass

        def load(self):
            raise RuntimeError("Extraction failed")

    monkeypatch.setattr(
        document_loader,
        "DoclingLoader",
        MockLoader
    )

    with pytest.raises(document_loader.DocumentLoadingError):

        document_loader.load_pdf_documents(pdf_path)
