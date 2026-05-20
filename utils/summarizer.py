"""Gemini-powered summarization for Gujarati newspaper text."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv


DEFAULT_MODEL = "gemini-1.5-flash"
DEFAULT_CHUNK_SIZE = 12_000
DEFAULT_OVERLAP = 500

LANGUAGE_LABELS = {
    "gujarati": "Gujarati",
    "hindi": "Hindi",
    "english": "English",
}


class SummarizationError(RuntimeError):
    """Raised when Gemini cannot create a summary."""


@dataclass(frozen=True)
class SummaryResult:
    """Structured summary output returned to the API and UI."""

    summary_text: str
    language: str
    model: str
    chunk_count: int


def _chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split long newspaper text into overlapping chunks for Gemini."""
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        boundary = text.rfind("\n", start, end)
        if boundary > start + chunk_size // 2:
            end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - overlap, 0)

    return chunks


def _build_prompt(text: str, language: str) -> str:
    return f"""
Summarize the following Gujarati newspaper content.
Provide:
1. A short summary
2. 5 important news bullet points
3. Likely news categories covered, such as politics, business, sports, local news, crime, entertainment, education, or weather

Use simple {language} language. Keep names, places, and numbers accurate.

Text:
{text}
""".strip()


def _build_final_prompt(chunk_summaries: Iterable[str], language: str) -> str:
    combined = "\n\n---\n\n".join(chunk_summaries)
    return f"""
Combine these partial Gujarati newspaper summaries into one clean final output.
Provide:
1. A short summary
2. 5 important news bullet points
3. Likely news categories covered

Use simple {language} language. Avoid repeating the same news point.

Partial summaries:
{combined}
""".strip()


def _get_gemini_model(model_name: str):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SummarizationError("GEMINI_API_KEY is missing. Add it to your .env file.")

    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise SummarizationError(
            "google-generativeai is not installed. Install requirements.txt first."
        ) from exc

    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(model_name)
    except Exception as exc:
        raise SummarizationError("Could not configure the Gemini API client.") from exc


def summarize_newspaper(
    text: str,
    *,
    target_language: str = "gujarati",
    model_name: str | None = None,
) -> SummaryResult:
    """Generate a short summary, five bullet points, and categories with Gemini."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise SummarizationError("Cannot summarize empty text.")

    normalized_language = target_language.lower().strip()
    language = LANGUAGE_LABELS.get(normalized_language, "Gujarati")
    model = model_name or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    chunks = _chunk_text(cleaned_text)
    gemini_model = _get_gemini_model(model)

    try:
        if len(chunks) == 1:
            response = gemini_model.generate_content(_build_prompt(chunks[0], language))
            summary_text = (response.text or "").strip()
        else:
            partial_summaries: list[str] = []
            for chunk in chunks:
                response = gemini_model.generate_content(_build_prompt(chunk, language))
                chunk_summary = (response.text or "").strip()
                if chunk_summary:
                    partial_summaries.append(chunk_summary)

            if not partial_summaries:
                raise SummarizationError("Gemini returned empty summaries for all chunks.")

            final_response = gemini_model.generate_content(
                _build_final_prompt(partial_summaries, language)
            )
            summary_text = (final_response.text or "").strip()
    except SummarizationError:
        raise
    except Exception as exc:
        raise SummarizationError("Gemini API failed while generating the summary.") from exc

    if not summary_text:
        raise SummarizationError("Gemini returned an empty summary.")

    return SummaryResult(
        summary_text=summary_text,
        language=language,
        model=model,
        chunk_count=len(chunks),
    )
