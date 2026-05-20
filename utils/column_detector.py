"""OpenCV column detection for newspaper article images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MAX_PROCESSING_SIDE = 1800
MIN_COLUMN_WIDTH_RATIO = 0.12
MAX_COLUMNS = 4


class ColumnDetectionError(RuntimeError):
    """Raised when a newspaper image cannot be segmented into columns."""


@dataclass(frozen=True)
class ColumnRegion:
    """A detected text column in left-to-right reading order."""

    column_id: int
    bbox: tuple[int, int, int, int]
    crop_path: str
    width: int
    height: int


def resize_for_processing(image: Image.Image) -> Image.Image:
    """Resize very large images while preserving aspect ratio for fast OCR."""
    image = image.convert("RGB")
    width, height = image.size
    max_side = max(width, height)
    if max_side <= MAX_PROCESSING_SIDE:
        return image

    scale = MAX_PROCESSING_SIDE / max_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def pil_to_cv(image: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV RGB array."""
    return np.array(image.convert("RGB"))


def _preprocess_for_columns(image: np.ndarray) -> np.ndarray:
    """Create a clean binary text mask for column projection analysis."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        15,
    )

    # Remove tiny speckles while keeping text strokes.
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, small_kernel, iterations=1)
    return binary


def _content_bounds(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Find the main newspaper content bounds, excluding empty margins."""
    points = cv2.findNonZero(binary)
    if points is None:
        raise ColumnDetectionError("No readable content was found in the image.")

    x, y, width, height = cv2.boundingRect(points)
    pad_x = max(4, int(width * 0.01))
    pad_y = max(4, int(height * 0.01))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(binary.shape[1], x + width + pad_x)
    bottom = min(binary.shape[0], y + height + pad_y)
    return left, top, right, bottom


def _smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    """Smooth a one-dimensional projection profile."""
    window = max(5, window)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(profile, kernel, mode="same")


def _find_whitespace_gutters(
    binary: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    """Detect vertical whitespace gutters between newspaper columns."""
    left, top, right, bottom = bounds
    content = binary[top:bottom, left:right]
    height, width = content.shape
    if width <= 0 or height <= 0:
        return []

    # Close text horizontally inside each column but keep gutters mostly empty.
    kernel_width = max(5, int(width * 0.006))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    connected = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel, iterations=1)

    ink_density = (connected > 0).sum(axis=0).astype(np.float32) / max(height, 1)
    smoothed = _smooth_profile(ink_density, max(9, int(width * 0.018)))
    threshold = max(0.012, min(0.07, float(np.percentile(smoothed, 25)) * 1.4))
    whitespace = smoothed <= threshold

    min_gap = max(14, int(width * 0.025))
    gutters: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_blank in enumerate(whitespace):
        if is_blank and start is None:
            start = index
        elif not is_blank and start is not None:
            if index - start >= min_gap:
                gutters.append((left + start, left + index))
            start = None
    if start is not None and width - start >= min_gap:
        gutters.append((left + start, left + width))

    # Drop outer margin gaps and keep strongest internal separators.
    content_width = right - left
    internal: list[tuple[int, int]] = []
    for gap_left, gap_right in gutters:
        center = (gap_left + gap_right) / 2
        if center < left + content_width * 0.08 or center > right - content_width * 0.08:
            continue
        internal.append((gap_left, gap_right))

    # Limit to 3 separators => up to 4 columns.
    internal = sorted(internal, key=lambda gap: gap[1] - gap[0], reverse=True)[: MAX_COLUMNS - 1]
    return sorted(internal, key=lambda gap: gap[0])


def _segments_from_gutters(
    bounds: tuple[int, int, int, int],
    gutters: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Build column rectangles from content bounds and whitespace gutters."""
    left, top, right, bottom = bounds
    segments: list[tuple[int, int, int, int]] = []
    current_left = left

    for gap_left, gap_right in gutters:
        if gap_left > current_left:
            segments.append((current_left, top, gap_left, bottom))
        current_left = gap_right
    if current_left < right:
        segments.append((current_left, top, right, bottom))

    min_width = max(40, int((right - left) * MIN_COLUMN_WIDTH_RATIO))
    return [segment for segment in segments if segment[2] - segment[0] >= min_width]


def _fallback_equal_columns(
    bounds: tuple[int, int, int, int],
    binary: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    """Pick a sensible equal-column fallback when gutters are weak."""
    left, top, right, bottom = bounds
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return [bounds]

    # Estimate columns by aspect ratio; newspapers/articles are often tall.
    aspect = width / max(height, 1)
    if aspect > 1.15:
        column_count = 4
    elif aspect > 0.85:
        column_count = 3
    elif aspect > 0.45:
        column_count = 2
    else:
        column_count = 1

    column_count = min(MAX_COLUMNS, max(1, column_count))
    if column_count == 1:
        return [bounds]

    column_width = width // column_count
    segments: list[tuple[int, int, int, int]] = []
    for index in range(column_count):
        seg_left = left + index * column_width
        seg_right = right if index == column_count - 1 else left + (index + 1) * column_width
        crop = binary[top:bottom, seg_left:seg_right]
        if (crop > 0).sum() < crop.size * 0.003:
            continue
        segments.append((seg_left, top, seg_right, bottom))

    return segments or [bounds]


def detect_columns(
    image: Image.Image,
    output_dir: str | Path,
) -> tuple[Image.Image, list[ColumnRegion]]:
    """
    Detect newspaper columns and save each crop.

    Returns the resized processing image and column regions sorted left-to-right.
    OCR should process the returned regions in list order.
    """
    processing_image = resize_for_processing(image)
    cv_image = pil_to_cv(processing_image)
    binary = _preprocess_for_columns(cv_image)
    bounds = _content_bounds(binary)
    gutters = _find_whitespace_gutters(binary, bounds)
    segments = _segments_from_gutters(bounds, gutters)
    if len(segments) <= 1:
        segments = _fallback_equal_columns(bounds, binary)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    regions: list[ColumnRegion] = []
    for index, (left, top, right, bottom) in enumerate(sorted(segments, key=lambda s: s[0]), start=1):
        crop = cv_image[top:bottom, left:right]
        if crop.size == 0:
            continue
        crop_path = output_path / f"column_{index:02d}.png"
        cv2.imwrite(str(crop_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        regions.append(
            ColumnRegion(
                column_id=len(regions) + 1,
                bbox=(left, top, right, bottom),
                crop_path=str(crop_path),
                width=right - left,
                height=bottom - top,
            )
        )

    if not regions:
        raise ColumnDetectionError("No valid text columns were detected.")

    return processing_image, regions
