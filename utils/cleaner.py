"""Gujarati OCR text cleaning helpers."""

from __future__ import annotations

import re
from collections import OrderedDict


_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")
_OCR_NOISE_RE = re.compile(r"[^\w\s\u0A80-\u0AFF.,;:!?()\-/\"'।॥%₹&+]")


def normalize_spaces(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    lines: list[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


def remove_duplicate_lines(text: str) -> str:
    """Remove repeated OCR lines such as running headers and page footers."""
    seen: OrderedDict[str, str] = OrderedDict()
    output: list[str] = []

    for line in text.splitlines():
        key = _WHITESPACE_RE.sub(" ", line).strip().lower()
        if not key:
            if output and output[-1] != "":
                output.append("")
            continue
        if key in seen:
            continue
        seen[key] = line.strip()
        output.append(line.strip())

    while output and output[-1] == "":
        output.pop()

    return "\n".join(output)


def clean_text(text: str) -> str:
    """Clean Gujarati OCR text without removing Gujarati Unicode characters."""
    if not text:
        return ""

    cleaned = text.replace("\x00", " ")
    cleaned = _OCR_NOISE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"([.,;:!?।॥]){3,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?।॥])", r"\1", cleaned)
    cleaned = normalize_spaces(cleaned)
    cleaned = remove_duplicate_lines(cleaned)
    cleaned = _MULTIPLE_NEWLINES_RE.sub("\n\n", cleaned)
    return cleaned.strip()
