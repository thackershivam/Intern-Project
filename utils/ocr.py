"""Selected-article OCR using PaddleOCR."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np

from utils.cleaner import clean_text


DEFAULT_OCR_LANGUAGE = "gu"


class OCRProcessingError(RuntimeError):
    """Raised when OCR cannot extract text from an article crop."""


@lru_cache(maxsize=2)
def _load_ocr_model(language: str = DEFAULT_OCR_LANGUAGE) -> Any:
    """Load PaddleOCR once per language for better UI performance."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRProcessingError(
            "PaddleOCR is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        return PaddleOCR(use_angle_cls=True, lang=language)
    except Exception as exc:
        raise OCRProcessingError(f"Could not initialize PaddleOCR language '{language}'.") from exc


def _extract_line_text(item: Any) -> str | None:
    """Read one recognized line from classic PaddleOCR result format."""
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


def _extract_text_from_result(result: Any) -> str:
    """Support common PaddleOCR result shapes across versions."""
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
    return "\n".join(lines)


def preprocess_article_for_ocr(image_path: str | Path) -> Path:
    """Create a cleaned, high-contrast temporary image for OCR."""
    source = Path(image_path)
    image = cv2.imread(str(source))
    if image is None:
        raise OCRProcessingError(f"Could not read article image: {source}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 1.6 if max(gray.shape) < 1600 else 1.0
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    temp = NamedTemporaryFile(prefix="article_ocr_", suffix=".png", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    cv2.imwrite(str(temp_path), thresholded)
    return temp_path


def extract_gujarati_text_from_article(
    article_image_path: str | Path,
    *,
    language: str = DEFAULT_OCR_LANGUAGE,
) -> str:
    """Run OCR only on the user-selected article crop."""
    preprocessed_path: Path | None = None
    try:
        preprocessed_path = preprocess_article_for_ocr(article_image_path)
        ocr = _load_ocr_model(language)
        result = ocr.ocr(str(preprocessed_path), cls=True)
        text = clean_text(_extract_text_from_result(result))
    except OCRProcessingError:
        raise
    except Exception as exc:
        raise OCRProcessingError("PaddleOCR failed while reading the selected article.") from exc
    finally:
        if preprocessed_path and preprocessed_path.exists():
            preprocessed_path.unlink(missing_ok=True)

    if not text:
        raise OCRProcessingError("No readable Gujarati text was found in the selected article.")

    return text
