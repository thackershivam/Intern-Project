"""OCR text cleaning for Gujarati, Hindi, and English."""

from __future__ import annotations

import re


_SPACE_RE = re.compile(r"[ \t\f\v]+")
_NOISE_RE = re.compile(
    r"[^\w\s\u0A80-\u0AFF\u0900-\u097F.,;:!?()\-/\"'।॥%₹&+]"
)


def clean_text(text: str) -> str:
    """Remove common OCR noise while preserving Indic scripts and paragraph order."""
    if not text:
        return ""

    cleaned = text.replace("\x00", " ")
    cleaned = _NOISE_RE.sub(" ", cleaned)
    cleaned = re.sub(r"([.,;:!?।॥]){3,}", r"\1", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?।॥])", r"\1", cleaned)

    lines: list[str] = []
    for raw_line in cleaned.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        line = _SPACE_RE.sub(" ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        lines.append(line)

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines).strip()
