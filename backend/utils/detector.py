"""OpenCV article block detector for interactive newspaper highlighting."""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MAX_IMAGE_SIDE = 1800
MAX_ARTICLES = 36
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


class DetectionError(RuntimeError):
    """Raised when a newspaper image cannot be processed."""


@dataclass(frozen=True)
class ArticleBox:
    """Detected article boundary and crop metadata."""

    id: int
    x: int
    y: int
    w: int
    h: int
    crop_path: str
    confidence: float

    def public_dict(self) -> dict[str, int | float]:
        """Return only browser-safe data."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "confidence": self.confidence,
        }

    def to_dict(self) -> dict[str, int | float | str]:
        """Return complete backend metadata."""
        return asdict(self)


def validate_extension(filename: str) -> None:
    """Validate supported upload extension."""
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DetectionError("Only JPEG, PNG, and PDF files are supported.")


def load_upload_image(upload_path: str | Path) -> Image.Image:
    """Load an uploaded image or convert the first PDF page."""
    path = Path(upload_path)
    validate_extension(path.name)

    if path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise DetectionError("pdf2image is not installed.") from exc

        try:
            pages = convert_from_path(
                str(path),
                dpi=180,
                first_page=1,
                last_page=1,
                fmt="png",
                poppler_path=os.getenv("POPPLER_PATH") or None,
            )
        except Exception as exc:
            raise DetectionError(
                "Could not convert PDF. Install Poppler or set POPPLER_PATH."
            ) from exc

        if not pages:
            raise DetectionError("The uploaded PDF has no pages.")
        return pages[0].convert("RGB")

    try:
        with Image.open(path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise DetectionError("Could not open uploaded newspaper image.") from exc


def resize_image(image: Image.Image) -> Image.Image:
    """Resize oversized scans for responsive frontend and faster detection."""
    image = image.convert("RGB")
    width, height = image.size
    max_side = max(width, height)
    if max_side <= MAX_IMAGE_SIDE:
        return image

    scale = MAX_IMAGE_SIDE / max_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _binary_text_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Create a binary mask for dark newspaper content."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel, iterations=1)


def _content_bounds(binary: np.ndarray) -> tuple[int, int, int, int]:
    points = cv2.findNonZero(binary)
    if points is None:
        raise DetectionError("No readable newspaper content was found.")

    x, y, w, h = cv2.boundingRect(points)
    pad_x = max(8, int(w * 0.015))
    pad_y = max(8, int(h * 0.015))
    return (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(binary.shape[1], x + w + pad_x),
        min(binary.shape[0], y + h + pad_y),
    )


def _is_page_border(box: tuple[int, int, int, int], image_shape: tuple[int, int]) -> bool:
    x, y, w, h = box
    height, width = image_shape
    return (
        x < width * 0.025
        and y < height * 0.025
        and w > width * 0.90
        and h > height * 0.90
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    """Remove heavily overlapping duplicate detections."""
    selected: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if any(_iou(box, existing) > 0.35 for existing in selected):
            continue
        selected.append(box)
    return selected


def _detect_boxes(binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect article-like regions with multiple morphology scales."""
    height, width = binary.shape
    page_area = height * width
    candidates: list[tuple[int, int, int, int]] = []

    # Multiple kernels help detect full articles in 2-4 column layouts while
    # keeping small logos/icons from becoming selectable article blocks.
    kernel_specs = (
        (0.018, 0.010, 2),
        (0.035, 0.018, 2),
        (0.055, 0.028, 1),
    )

    for width_ratio, height_ratio, iterations in kernel_specs:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (max(9, int(width * width_ratio)), max(5, int(height * height_ratio))),
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            box_area = w * h
            contour_area = cv2.contourArea(contour)
            aspect = w / max(h, 1)

            if _is_page_border((x, y, w, h), (height, width)):
                continue
            if w < width * 0.08 or h < height * 0.035:
                continue
            if box_area < page_area * 0.006 or box_area > page_area * 0.75:
                continue
            if contour_area / max(box_area, 1) < 0.035:
                continue
            if aspect < 0.12 or aspect > 8.0:
                continue

            candidates.append((x, y, w, h))

    return _dedupe_boxes(candidates)


def _fallback_column_boxes(bounds: tuple[int, int, int, int], binary: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback for clean column layouts where article grouping is ambiguous."""
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    aspect = width / max(height, 1)
    if aspect > 1.15:
        count = 4
    elif aspect > 0.85:
        count = 3
    elif aspect > 0.42:
        count = 2
    else:
        count = 1

    column_width = width // count
    boxes: list[tuple[int, int, int, int]] = []
    for index in range(count):
        x1 = left + index * column_width
        x2 = right if index == count - 1 else left + (index + 1) * column_width
        crop = binary[top:bottom, x1:x2]
        if crop.size and (crop > 0).sum() > crop.size * 0.002:
            boxes.append((x1, top, x2 - x1, height))
    return boxes or [(left, top, width, height)]


def _sort_reading_order(boxes: list[tuple[int, int, int, int]], page_width: int) -> list[tuple[int, int, int, int]]:
    """Sort by article location in a stable newspaper reading order."""
    # Group by rough columns first, then top-to-bottom inside each column band.
    def key(box: tuple[int, int, int, int]) -> tuple[int, int, int]:
        x, y, w, _ = box
        center = x + w // 2
        band = int(center / max(1, page_width / 6))
        return (band, y, x)

    return sorted(boxes, key=key)


def detect_articles(
    image: Image.Image,
    crop_dir: str | Path,
) -> tuple[Image.Image, list[ArticleBox]]:
    """Detect article regions, save crops, and return frontend bounding boxes."""
    processed_image = resize_image(image)
    image_rgb = np.array(processed_image.convert("RGB"))
    height, width = image_rgb.shape[:2]
    binary = _binary_text_mask(image_rgb)
    bounds = _content_bounds(binary)
    boxes = _detect_boxes(binary)

    if len(boxes) < 2:
        boxes = _fallback_column_boxes(bounds, binary)

    boxes = _sort_reading_order(boxes, width)[:MAX_ARTICLES]
    destination = Path(crop_dir)
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    destination.mkdir(parents=True, exist_ok=True)

    articles: list[ArticleBox] = []
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        pad = max(6, int(min(width, height) * 0.005))
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(width, x + w + pad)
        bottom = min(height, y + h + pad)
        if right <= left or bottom <= top:
            continue

        crop = image_rgb[top:bottom, left:right]
        crop_path = destination / f"article_{index:02d}.png"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        area_ratio = ((right - left) * (bottom - top)) / max(width * height, 1)
        confidence = max(0.05, min(0.99, 0.35 + area_ratio * 3))
        articles.append(
            ArticleBox(
                id=len(articles) + 1,
                x=left,
                y=top,
                w=right - left,
                h=bottom - top,
                crop_path=str(crop_path),
                confidence=round(confidence, 3),
            )
        )

    if not articles:
        raise DetectionError("No article regions were detected.")

    return processed_image, articles
