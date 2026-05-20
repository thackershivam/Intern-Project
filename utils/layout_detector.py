"""OpenCV-based newspaper article layout detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MAX_WORKING_SIDE = 1400
DEFAULT_MAX_ARTICLES = 30


class LayoutDetectionError(RuntimeError):
    """Raised when article regions cannot be detected."""


@dataclass(frozen=True)
class ArticleRegion:
    """Detected newspaper article region."""

    article_id: int
    bbox: tuple[int, int, int, int]
    crop_path: str
    width: int
    height: int
    area: int
    confidence: float

    def to_dict(self) -> dict[str, int | float | str | tuple[int, int, int, int]]:
        """Return a JSON-friendly dictionary."""
        return asdict(self)


def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to an OpenCV RGB array."""
    return np.array(image.convert("RGB"))


def cv_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV RGB array to a PIL image."""
    return Image.fromarray(image.astype("uint8"), mode="RGB")


def _resize_for_detection(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    max_side = max(height, width)
    if max_side <= MAX_WORKING_SIDE:
        return image.copy(), 1.0

    scale = MAX_WORKING_SIDE / max_side
    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _remove_page_border(box: tuple[int, int, int, int], image_shape: tuple[int, int]) -> bool:
    x, y, w, h = box
    image_height, image_width = image_shape
    covers_width = w > image_width * 0.92
    covers_height = h > image_height * 0.92
    touches_edges = x < image_width * 0.025 and y < image_height * 0.025
    return bool(covers_width and covers_height and touches_edges)


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def _merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
    *,
    iou_threshold: float = 0.25,
) -> list[tuple[int, int, int, int]]:
    """Merge duplicate or highly overlapping article boxes."""
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda item: item[2] * item[3], reverse=True):
        if any(_iou(box, existing) > iou_threshold for existing in merged):
            continue
        merged.append(box)
    return merged


def _score_box(box: tuple[int, int, int, int], image_shape: tuple[int, int]) -> float:
    """Score how article-like a bounding box is."""
    _, _, width, height = box
    image_height, image_width = image_shape
    area_ratio = (width * height) / (image_width * image_height)
    aspect_ratio = width / max(height, 1)

    area_score = min(area_ratio / 0.08, 1.0)
    aspect_penalty = 0.0
    if aspect_ratio < 0.18 or aspect_ratio > 6.0:
        aspect_penalty = 0.35

    return max(0.05, min(0.99, 0.35 + area_score * 0.6 - aspect_penalty))


def _detect_candidate_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect possible article blocks with thresholding and morphology."""
    working_image, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(working_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )

    image_height, image_width = binary.shape
    page_area = image_height * image_width
    boxes: list[tuple[int, int, int, int]] = []

    # Kernels are tuned to connect Gujarati text lines into article/column blocks
    # while avoiding one huge whole-page component.
    kernel_specs = (
        (0.020, 0.010, 2),
        (0.035, 0.018, 2),
        (0.055, 0.026, 1),
    )
    for width_ratio, height_ratio, iterations in kernel_specs:
        kernel_width = max(9, int(image_width * width_ratio))
        kernel_height = max(5, int(image_height * height_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        closed = cv2.dilate(closed, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            contour_area = cv2.contourArea(contour)
            box_area = width * height

            if _remove_page_border((x, y, width, height), (image_height, image_width)):
                continue
            if width < image_width * 0.08 or height < image_height * 0.035:
                continue
            if box_area < page_area * 0.006 or box_area > page_area * 0.80:
                continue
            if contour_area / max(box_area, 1) < 0.04:
                continue

            original_x = int(x / scale)
            original_y = int(y / scale)
            original_w = int(width / scale)
            original_h = int(height / scale)
            boxes.append((original_x, original_y, original_w, original_h))

    return _merge_overlapping_boxes(boxes)


def _fallback_column_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Fallback for pages where morphology detects too few article boxes."""
    height, width = image.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    column_count = 3 if width > height * 0.65 else 2
    margin_x = int(width * 0.04)
    margin_y = int(height * 0.04)
    usable_width = width - margin_x * 2
    column_width = usable_width // column_count

    for column_index in range(column_count):
        x = margin_x + column_index * column_width
        w = column_width - int(width * 0.015)
        boxes.append((x, margin_y, max(1, w), max(1, height - margin_y * 2)))

    return boxes


def detect_articles(
    image: Image.Image,
    output_dir: str | Path,
    *,
    max_articles: int = DEFAULT_MAX_ARTICLES,
) -> list[ArticleRegion]:
    """Detect, crop, and save article-like regions from a newspaper image."""
    cv_image = pil_to_cv(image)
    image_height, image_width = cv_image.shape[:2]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    boxes = _detect_candidate_boxes(cv_image)
    if len(boxes) < 2:
        boxes = _fallback_column_boxes(cv_image)

    boxes = sorted(
        boxes,
        key=lambda box: (box[1] // max(1, image_height // 20), box[0], box[1]),
    )[:max_articles]

    regions: list[ArticleRegion] = []
    for index, (x, y, width, height) in enumerate(boxes, start=1):
        padding = max(8, int(min(image_width, image_height) * 0.006))
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(image_width, x + width + padding)
        bottom = min(image_height, y + height + padding)
        if right <= left or bottom <= top:
            continue

        crop = cv_image[top:bottom, left:right]
        crop_file = output_path / f"article_{index:02d}.png"
        cv2.imwrite(str(crop_file), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        region_width = right - left
        region_height = bottom - top
        regions.append(
            ArticleRegion(
                article_id=len(regions) + 1,
                bbox=(left, top, right, bottom),
                crop_path=str(crop_file),
                width=region_width,
                height=region_height,
                area=region_width * region_height,
                confidence=_score_box((left, top, region_width, region_height), (image_height, image_width)),
            )
        )

    if not regions:
        raise LayoutDetectionError("No article regions were detected in this newspaper.")

    return regions
