"""Visualization helpers for detected newspaper article regions."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.layout_detector import ArticleRegion, pil_to_cv


BOX_COLOR = (0, 132, 255)
SELECTED_COLOR = (0, 190, 70)
TEXT_COLOR = (255, 255, 255)


def draw_article_boxes(
    image: Image.Image,
    regions: list[ArticleRegion],
    *,
    selected_article_id: int | None = None,
) -> Image.Image:
    """Draw numbered article boundaries on the newspaper image."""
    canvas = pil_to_cv(image).copy()

    for region in regions:
        left, top, right, bottom = region.bbox
        is_selected = region.article_id == selected_article_id
        color = SELECTED_COLOR if is_selected else BOX_COLOR
        thickness = 5 if is_selected else 3

        overlay = canvas.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), color, -1)
        canvas = cv2.addWeighted(overlay, 0.08 if not is_selected else 0.14, canvas, 0.92, 0)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness)

        label = f"{region.article_id}"
        font_scale = max(0.8, min(canvas.shape[:2]) / 1000)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_width, label_height = label_size
        label_right = left + label_width + 18
        label_bottom = top + label_height + 18
        cv2.rectangle(canvas, (left, top), (label_right, label_bottom), color, -1)
        cv2.putText(
            canvas,
            label,
            (left + 8, top + label_height + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )

    return Image.fromarray(canvas)


def save_visualization(
    image: Image.Image,
    regions: list[ArticleRegion],
    output_path: str | Path,
    *,
    selected_article_id: int | None = None,
) -> Path:
    """Save a newspaper image with article boundaries."""
    visualized = draw_article_boxes(
        image,
        regions,
        selected_article_id=selected_article_id,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    visualized.save(destination)
    return destination


def create_thumbnail(image_path: str | Path, max_size: tuple[int, int] = (320, 220)) -> Image.Image:
    """Create a preview thumbnail for an article crop."""
    with Image.open(image_path) as image:
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail(max_size)
        return thumbnail.copy()
