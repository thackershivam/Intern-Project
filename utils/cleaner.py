"""Text cleaning helpers for newspaper extraction output."""

from __future__ import annotations

import re
from collections import OrderedDict


_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_OCR_NOISE_RE = re.compile(r"[^\w\s\u0A80-\u0AFF.,;:!?()\-/\"'।॥%₹&+]")


def normalize_spaces(text: str) -> str:
    """Collapse noisy spacing while preserving meaningful line breaks."""
    normalized_lines: list[str] = []
    for line in text.splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        if line:
            normalized_lines.append(line)
    return "\n".join(normalized_lines)


def remove_duplicate_lines(text: str) -> str:
    """Remove repeated OCR/header/footer lines while keeping original order."""
    unique_lines: OrderedDict[str, str] = OrderedDict()
    for line in text.splitlines():
        normalized_key = _WHITESPACE_RE.sub(" ", line).strip().lower()
        if not normalized_key:
            continue
        unique_lines.setdefault(normalized_key, line.strip())
    return "\n".join(unique_lines.values())


def clean_ocr_noise(text: str) -> str:
    """Remove common OCR artifacts without stripping Gujarati characters."""
    text = text.replace("\x00", " ")
    text = _OCR_NOISE_RE.sub(" ", text)
    text = re.sub(r"([.,;:!?।॥]){3,}", r"\1", text)
    text = re.sub(r"\s+([.,;:!?।॥])", r"\1", text)
    return text


def clean_text(text: str) -> str:
    """Run all cleaning steps for extracted newspaper text."""
    if not text:
        return ""

    cleaned = clean_ocr_noise(text)
    cleaned = normalize_spaces(cleaned)
    cleaned = remove_duplicate_lines(cleaned)
    cleaned = _MULTIPLE_NEWLINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()
