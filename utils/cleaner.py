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
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = _WHITESPACE_RE.sub(" ", line).strip()
        if not line:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        normalized_lines.append(line)

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def remove_duplicate_lines(text: str) -> str:
    """Remove repeated OCR/header/footer lines while keeping original order."""
    unique_lines: OrderedDict[str, str] = OrderedDict()
    output_lines: list[str] = []
    for line in text.splitlines():
        normalized_key = _WHITESPACE_RE.sub(" ", line).strip().lower()
        if not normalized_key:
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            continue
        if normalized_key in unique_lines:
            continue
        unique_lines[normalized_key] = line.strip()
        output_lines.append(line.strip())

    while output_lines and output_lines[-1] == "":
        output_lines.pop()

    return "\n".join(output_lines)


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
