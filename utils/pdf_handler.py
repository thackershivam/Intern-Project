"""PDF and image upload helpers."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


class InputFileError(RuntimeError):
    """Raised when an uploaded newspaper file cannot be loaded."""


def is_supported_file(filename: str) -> bool:
    """Return True when the upload extension is supported."""
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS


def is_pdf(path: str | Path) -> bool:
    """Return True when the path points to a PDF."""
    return Path(path).suffix.lower() == ".pdf"


def load_newspaper_image(path: str | Path) -> Image.Image:
    """Load an uploaded image or convert the first PDF page to a PIL image."""
    upload_path = Path(path)
    if not upload_path.exists():
        raise InputFileError(f"Uploaded file not found: {upload_path}")
    if upload_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise InputFileError("Only JPEG, PNG, and PDF files are supported.")

    if is_pdf(upload_path):
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise InputFileError(
                "pdf2image is not installed. Run `pip install -r requirements.txt`."
            ) from exc

        poppler_path = os.getenv("POPPLER_PATH") or None
        try:
            pages = convert_from_path(
                str(upload_path),
                dpi=180,
                first_page=1,
                last_page=1,
                fmt="png",
                poppler_path=poppler_path,
            )
        except Exception as exc:
            raise InputFileError(
                "Could not convert PDF first page. Check Poppler installation or POPPLER_PATH."
            ) from exc

        if not pages:
            raise InputFileError("No pages were found in this PDF.")
        return pages[0].convert("RGB")

    try:
        with Image.open(upload_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise InputFileError("Could not open uploaded newspaper image.") from exc
