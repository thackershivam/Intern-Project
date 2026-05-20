"""PDF text extraction with pdfplumber first and OCR fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.cleaner import clean_text
from utils.ocr import OCRProcessingError, extract_text_with_ocr


MIN_EXTRACTED_CHARS = 500


class PDFExtractionError(RuntimeError):
    """Raised when text cannot be extracted from an uploaded PDF."""


@dataclass(frozen=True)
class ExtractionResult:
    """Metadata and cleaned text extracted from a PDF."""

    text: str
    method: str
    page_count: int


def validate_pdf(pdf_path: str | Path) -> None:
    """Validate the uploaded file path and basic PDF signature."""
    pdf = Path(pdf_path)
    if not pdf.exists() or not pdf.is_file():
        raise PDFExtractionError("Uploaded PDF file was not found.")
    if pdf.suffix.lower() != ".pdf":
        raise PDFExtractionError("Only PDF files are supported.")

    try:
        with pdf.open("rb") as file:
            signature = file.read(5)
    except OSError as exc:
        raise PDFExtractionError("Could not read the uploaded PDF.") from exc

    if signature != b"%PDF-":
        raise PDFExtractionError("The uploaded file is not a valid PDF.")


def extract_text_with_pdfplumber(pdf_path: str | Path) -> tuple[str, int]:
    """Extract selectable text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError as exc:
        raise PDFExtractionError(
            "pdfplumber is not installed. Install requirements.txt before running extraction."
        ) from exc

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_text: list[str] = []
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                if text.strip():
                    page_text.append(text)
            return "\n\n".join(page_text), len(pdf.pages)
    except Exception as exc:
        raise PDFExtractionError("pdfplumber could not read this PDF.") from exc


def extract_text_from_pdf(
    pdf_path: str | Path,
    *,
    min_chars: int = MIN_EXTRACTED_CHARS,
    ocr_language: str = "gu",
) -> ExtractionResult:
    """Extract Gujarati text, falling back to PaddleOCR for scanned PDFs."""
    validate_pdf(pdf_path)

    plumber_text = ""
    page_count = 0
    try:
        plumber_text, page_count = extract_text_with_pdfplumber(pdf_path)
    except PDFExtractionError:
        # Scanned or damaged PDFs can fail in pdfplumber; OCR may still work.
        plumber_text = ""

    cleaned_plumber_text = clean_text(plumber_text)
    if len(cleaned_plumber_text) >= min_chars:
        return ExtractionResult(
            text=cleaned_plumber_text,
            method="pdfplumber",
            page_count=page_count,
        )

    try:
        ocr_text = extract_text_with_ocr(pdf_path, language=ocr_language)
    except OCRProcessingError as exc:
        if cleaned_plumber_text:
            return ExtractionResult(
                text=cleaned_plumber_text,
                method="pdfplumber_partial",
                page_count=page_count,
            )
        raise PDFExtractionError(str(exc)) from exc

    cleaned_ocr_text = clean_text(ocr_text)
    if not cleaned_ocr_text:
        raise PDFExtractionError("No readable Gujarati text was found in this PDF.")

    return ExtractionResult(
        text=cleaned_ocr_text,
        method="paddleocr",
        page_count=page_count,
    )
