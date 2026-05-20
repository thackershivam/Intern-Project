"""Paragraph-based article separation for extracted newspaper text."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


MIN_ARTICLE_CHARS = 80
MIN_HEADING_CHARS = 4
TITLE_MAX_CHARS = 90


@dataclass(frozen=True)
class Article:
    """A paragraph-based article section extracted from the PDF text."""

    index: int
    title: str
    text: str
    char_count: int
    word_count: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _normalize_paragraph(paragraph: str) -> str:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _split_paragraphs(text: str) -> list[str]:
    """Split text by blank-line paragraphs, with line fallback for OCR output."""
    paragraphs = [
        _normalize_paragraph(part)
        for part in re.split(r"\n\s*\n+", text.strip())
        if _normalize_paragraph(part)
    ]

    if len(paragraphs) > 1:
        return paragraphs

    # Some PDF/OCR output loses blank lines. In that case, treat longer lines as
    # paragraph sections so the UI can still separate visible article blocks.
    return [
        _normalize_paragraph(line)
        for line in text.splitlines()
        if _normalize_paragraph(line)
    ]


def _build_title(article_text: str, index: int) -> str:
    first_line = next(
        (line.strip() for line in article_text.splitlines() if line.strip()),
        "",
    )
    title = first_line[:TITLE_MAX_CHARS].strip(" -:|")
    return title or f"Article {index}"


def separate_articles_by_paragraph(
    text: str,
    *,
    min_article_chars: int = MIN_ARTICLE_CHARS,
) -> list[Article]:
    """
    Separate extracted newspaper text into article sections using paragraphs.

    Short paragraphs are treated as possible headlines and attached to the next
    substantial paragraph. If the PDF contains only short paragraphs, they are
    still returned so the user can inspect all extracted content.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    articles: list[Article] = []
    pending_heading: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) < min_article_chars:
            if len(paragraph) >= MIN_HEADING_CHARS:
                pending_heading.append(paragraph)
            continue

        article_text_parts = [*pending_heading, paragraph]
        pending_heading = []
        article_text = "\n\n".join(article_text_parts).strip()
        articles.append(
            Article(
                index=len(articles) + 1,
                title=_build_title(article_text, len(articles) + 1),
                text=article_text,
                char_count=len(article_text),
                word_count=len(article_text.split()),
            )
        )

    if pending_heading:
        leftover_text = "\n\n".join(pending_heading).strip()
        articles.append(
            Article(
                index=len(articles) + 1,
                title=_build_title(leftover_text, len(articles) + 1),
                text=leftover_text,
                char_count=len(leftover_text),
                word_count=len(leftover_text.split()),
            )
        )

    if not articles:
        for paragraph in paragraphs:
            articles.append(
                Article(
                    index=len(articles) + 1,
                    title=_build_title(paragraph, len(articles) + 1),
                    text=paragraph,
                    char_count=len(paragraph),
                    word_count=len(paragraph.split()),
                )
            )

    return articles


def combine_articles_for_summary(articles: list[Article], fallback_text: str) -> str:
    """Create a Gemini-friendly text block with article labels."""
    if not articles:
        return fallback_text

    return "\n\n".join(
        f"Article {article.index}: {article.title}\n{article.text}"
        for article in articles
    )
