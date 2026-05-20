"""OCR text cleaning utilities."""

from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_OCR_NOISE_RE = re.compile(
    r"[^\w\s\u0A80-\u0AFF\u0900-\u097F.,;:!?()\-/\"'।॥%₹&+]"
)


def clean_text(text: str) -> str:
    """Clean OCR output while preserving Gujarati, Hindi, and English text."""
    if not text:
        return ""

    cleaned = text.replace("\x00", " ")
    cleaned = _OCR_NOISE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"([.,;:!?।॥]){3,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?।॥])", r"\1", cleaned)

    lines: list[str] = []
    for raw_line in cleaned.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()


def merge_column_texts(column_texts: list[str]) -> str:
    """Merge OCR text in strict left-to-right column reading order."""
    return "\n\n".join(text.strip() for text in column_texts if text.strip()).strip()
