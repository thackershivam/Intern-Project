"""Visualization helpers for detected newspaper columns."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.column_detector import ColumnRegion


BOX_COLOR = (0, 128, 255)
LABEL_COLOR = (0, 90, 200)
TEXT_COLOR = (255, 255, 255)


def draw_column_boxes(image: Image.Image, columns: list[ColumnRegion]) -> Image.Image:
    """Draw numbered column boundaries on the processed newspaper image."""
    canvas = np.array(image.convert("RGB")).copy()
    for column in columns:
        left, top, right, bottom = column.bbox
        overlay = canvas.copy()
        cv2.rectangle(overlay, (left, top), (right, bottom), BOX_COLOR, -1)
        canvas = cv2.addWeighted(overlay, 0.08, canvas, 0.92, 0)
        cv2.rectangle(canvas, (left, top), (right, bottom), BOX_COLOR, 4)

        label = f"Column {column.column_id}"
        font_scale = max(0.7, min(canvas.shape[:2]) / 1200)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        label_width, label_height = label_size
        cv2.rectangle(
            canvas,
            (left, top),
            (left + label_width + 18, top + label_height + 18),
            LABEL_COLOR,
            -1,
        )
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


def make_thumbnail(image_path: str | Path, size: tuple[int, int] = (280, 320)) -> Image.Image:
    """Create a thumbnail preview for a cropped column."""
    with Image.open(image_path) as image:
        thumbnail = image.convert("RGB")
        thumbnail.thumbnail(size)
        return thumbnail.copy()
