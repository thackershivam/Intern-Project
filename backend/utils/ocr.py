"""PaddleOCR pipeline for selected article crops."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2

from backend.utils.cleaner import clean_text


OCR_LANGUAGES = {
    "Gujarati": "gu",
    "Hindi": "hi",
    "English": "en",
}


class OCRError(RuntimeError):
    """Raised when OCR fails."""


def get_ocr_lang(language: str) -> str:
    """Map UI language label to PaddleOCR language code."""
    return OCR_LANGUAGES.get(language, "gu")


@lru_cache(maxsize=3)
def _load_ocr(lang: str) -> Any:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRError("PaddleOCR is not installed.") from exc

    try:
        return PaddleOCR(use_angle_cls=True, lang=lang)
    except Exception as exc:
        raise OCRError(f"Could not initialize PaddleOCR for language '{lang}'.") from exc


def _classic_line(item: Any) -> tuple[float, float, str] | None:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    box = item[0]
    recognition = item[1]
    if (
        not isinstance(recognition, (list, tuple))
        or not recognition
        or not isinstance(recognition[0], str)
    ):
        return None
    try:
        min_x = min(float(point[0]) for point in box)
        min_y = min(float(point[1]) for point in box)
    except Exception:
        min_x = 0.0
        min_y = 0.0
    return min_y, min_x, recognition[0]


def _extract_ordered_text(result: Any) -> str:
    """Extract OCR lines sorted top-to-bottom, left-to-right."""
    lines: list[tuple[float, float, str]] = []

    def walk(node: Any) -> None:
        if not node:
            return
        parsed = _classic_line(node)
        if parsed:
            lines.append(parsed)
            return
        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts")
            if isinstance(texts, list):
                lines.extend((float(index), 0.0, str(text)) for index, text in enumerate(texts) if text)
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    walk(value)
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(result)
    return "\n".join(text for _, _, text in sorted(lines, key=lambda row: (row[0], row[1])))


def _preprocess_for_ocr(image_path: str | Path) -> Path:
    """Enhance selected article crop before OCR."""
    source = Path(image_path)
    image = cv2.imread(str(source))
    if image is None:
        raise OCRError("Could not read selected article crop.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if max(gray.shape) < 1400:
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, h=8)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )

    temp_file = NamedTemporaryFile(prefix="article_ocr_", suffix=".png", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    cv2.imwrite(str(temp_path), thresholded)
    return temp_path


def extract_article_text(crop_path: str | Path, language: str) -> str:
    """Run OCR only on the clicked article crop."""
    preprocessed: Path | None = None
    try:
        preprocessed = _preprocess_for_ocr(crop_path)
        ocr = _load_ocr(get_ocr_lang(language))
        result = ocr.ocr(str(preprocessed), cls=True)
        text = clean_text(_extract_ordered_text(result))
    except OCRError:
        raise
    except Exception as exc:
        raise OCRError("PaddleOCR failed on the selected article.") from exc
    finally:
        if preprocessed:
            preprocessed.unlink(missing_ok=True)

    if not text:
        raise OCRError("No text was detected in the selected article.")

    return text
