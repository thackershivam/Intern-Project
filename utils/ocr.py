"""OCR support for scanned Gujarati newspaper PDFs."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


DEFAULT_OCR_LANGUAGE = "gu"


class OCRProcessingError(RuntimeError):
    """Raised when OCR extraction cannot complete."""


def _load_ocr_model(language: str = DEFAULT_OCR_LANGUAGE) -> Any:
    """Create a PaddleOCR instance lazily so the app can start quickly."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRProcessingError(
            "PaddleOCR is not installed. Install requirements.txt before using OCR."
        ) from exc

    try:
        return PaddleOCR(use_angle_cls=True, lang=language)
    except Exception as exc:  # PaddleOCR raises several framework-specific errors.
        raise OCRProcessingError(f"Unable to initialize PaddleOCR for language '{language}'.") from exc


def _extract_line_text(item: Any) -> str | None:
    """Extract text from one classic PaddleOCR detection line."""
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    recognition = item[1]
    if (
        isinstance(recognition, (list, tuple))
        and recognition
        and isinstance(recognition[0], str)
    ):
        return recognition[0]
    return None


def _extract_text_from_ocr_result(result: Any) -> str:
    """Handle PaddleOCR result shapes across common package versions."""
    lines: list[str] = []

    def walk(node: Any) -> None:
        if not node:
            return

        if isinstance(node, dict):
            for key in ("rec_texts", "texts"):
                values = node.get(key)
                if isinstance(values, list):
                    lines.extend(str(value) for value in values if value)
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    walk(value)
            return

        line_text = _extract_line_text(node)
        if line_text:
            lines.append(line_text)
            return

        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(result)
    return "\n".join(line for line in lines if line)


def extract_text_with_ocr(
    pdf_path: str | Path,
    *,
    language: str = DEFAULT_OCR_LANGUAGE,
    dpi: int = 220,
) -> str:
    """Convert PDF pages to images and extract Gujarati text with PaddleOCR."""
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise OCRProcessingError(f"PDF not found: {pdf}")

    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise OCRProcessingError(
            "pdf2image is not installed. Install requirements.txt before using OCR."
        ) from exc

    ocr = _load_ocr_model(language)
    extracted_pages: list[str] = []

    try:
        with TemporaryDirectory(prefix="gujarati_ocr_") as temp_dir:
            images = convert_from_path(
                str(pdf),
                dpi=dpi,
                output_folder=temp_dir,
                fmt="png",
                thread_count=2,
            )

            for page_number, image in enumerate(images, start=1):
                image_path = Path(temp_dir) / f"page_{page_number}.png"
                image.save(image_path, "PNG")
                result = ocr.ocr(str(image_path), cls=True)
                page_text = _extract_text_from_ocr_result(result)
                if page_text:
                    extracted_pages.append(page_text)
    except Exception as exc:
        raise OCRProcessingError("OCR failed while processing the PDF pages.") from exc

    return "\n\n".join(extracted_pages).strip()
