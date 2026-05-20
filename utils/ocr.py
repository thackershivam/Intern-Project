"""PaddleOCR helpers for column-wise newspaper OCR."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2

from utils.cleaner import clean_text
from utils.column_detector import ColumnRegion


LANGUAGE_CONFIG = {
    "Gujarati": {"ocr": "gu", "tts": "gu"},
    "Hindi": {"ocr": "hi", "tts": "hi"},
    "English": {"ocr": "en", "tts": "en"},
}


class OCRProcessingError(RuntimeError):
    """Raised when PaddleOCR cannot process a column."""


def get_ocr_language(language_label: str) -> str:
    """Return PaddleOCR language code for the selected UI language."""
    return LANGUAGE_CONFIG.get(language_label, LANGUAGE_CONFIG["Gujarati"])["ocr"]


@lru_cache(maxsize=3)
def _load_ocr_model(language_code: str) -> Any:
    """Load PaddleOCR once per language."""
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRProcessingError(
            "PaddleOCR is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        return PaddleOCR(use_angle_cls=True, lang=language_code)
    except Exception as exc:
        raise OCRProcessingError(
            f"Could not initialize PaddleOCR for language '{language_code}'."
        ) from exc


def _line_from_classic_result(item: Any) -> tuple[float, str] | None:
    """Extract y-position and text from classic PaddleOCR output."""
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
        y_position = min(float(point[1]) for point in box)
    except Exception:
        y_position = 0.0
    return y_position, recognition[0]


def _extract_ordered_text(result: Any) -> str:
    """Extract OCR text sorted top-to-bottom within one column."""
    lines: list[tuple[float, str]] = []

    if isinstance(result, dict):
        texts = result.get("rec_texts") or result.get("texts") or []
        if isinstance(texts, list):
            return "\n".join(str(text) for text in texts if text)

    def walk(node: Any) -> None:
        if not node:
            return

        parsed = _line_from_classic_result(node)
        if parsed:
            lines.append(parsed)
            return

        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts")
            if isinstance(texts, list):
                lines.extend((float(index), str(text)) for index, text in enumerate(texts) if text)
            for value in node.values():
                if isinstance(value, (dict, list, tuple)):
                    walk(value)
            return

        if isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(result)
    return "\n".join(text for _, text in sorted(lines, key=lambda line: line[0]))


def preprocess_column_for_ocr(column_path: str | Path) -> Path:
    """Enhance one column crop for OCR."""
    source = Path(column_path)
    image = cv2.imread(str(source))
    if image is None:
        raise OCRProcessingError(f"Could not read column image: {source}")

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

    temp_file = NamedTemporaryFile(prefix="column_ocr_", suffix=".png", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    cv2.imwrite(str(temp_path), thresholded)
    return temp_path


def extract_text_from_column(column: ColumnRegion, language_label: str) -> str:
    """Run PaddleOCR on one detected column crop."""
    language_code = get_ocr_language(language_label)
    preprocessed_path: Path | None = None
    try:
        preprocessed_path = preprocess_column_for_ocr(column.crop_path)
        ocr_model = _load_ocr_model(language_code)
        result = ocr_model.ocr(str(preprocessed_path), cls=True)
        return clean_text(_extract_ordered_text(result))
    except OCRProcessingError:
        raise
    except Exception as exc:
        raise OCRProcessingError(f"OCR failed for column {column.column_id}.") from exc
    finally:
        if preprocessed_path:
            preprocessed_path.unlink(missing_ok=True)


def extract_text_column_wise(
    columns: list[ColumnRegion],
    language_label: str,
    *,
    progress_callback=None,
) -> list[str]:
    """OCR each column in left-to-right order."""
    texts: list[str] = []
    for index, column in enumerate(sorted(columns, key=lambda item: item.bbox[0]), start=1):
        if progress_callback:
            progress_callback(index, len(columns), column)
        texts.append(extract_text_from_column(column, language_label))
    return texts
