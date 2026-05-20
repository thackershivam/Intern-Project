"""Image-based newspaper article cutouts from scanned PDF pages."""

from __future__ import annotations

import shutil
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


MAX_DETECTION_SIDE = 900
DEFAULT_DPI = 170
DEFAULT_MAX_CROPS_PER_PAGE = 24


class ArticleCropError(RuntimeError):
    """Raised when article image cutouts cannot be created."""


@dataclass(frozen=True)
class ArticleImageCrop:
    """Metadata for one cropped newspaper image block."""

    index: int
    page_number: int
    image_path: str
    bbox: tuple[int, int, int, int]
    width: int
    height: int

    def to_dict(self) -> dict[str, int | str | tuple[int, int, int, int]]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _make_odd(value: int) -> int:
    value = max(value, 3)
    return value if value % 2 else value + 1


def _resize_for_detection(image: Image.Image) -> tuple[Image.Image, float]:
    width, height = image.size
    max_side = max(width, height)
    if max_side <= MAX_DETECTION_SIDE:
        return image.copy(), 1.0

    scale = MAX_DETECTION_SIDE / max_side
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return resized, scale


def _content_mask(image: Image.Image, dilation_size: int) -> np.ndarray:
    """Create a binary mask of dark newspaper content."""
    grayscale = image.convert("L")
    pixels = np.asarray(grayscale)

    # Newspaper scans are usually light background with dark text/photos.
    threshold = min(245, max(190, int(np.percentile(pixels, 82))))
    mask = (pixels < threshold).astype("uint8") * 255
    mask_image = Image.fromarray(mask, mode="L")
    dilated = mask_image.filter(ImageFilter.MaxFilter(_make_odd(dilation_size)))
    return np.asarray(dilated) > 0


def _connected_component_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """Find connected components in a small binary mask."""
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []

    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue

            queue: deque[tuple[int, int]] = deque([(start_x, start_y)])
            visited[start_y, start_x] = True
            min_x = max_x = start_x
            min_y = max_y = start_y
            area = 0

            while queue:
                x, y = queue.popleft()
                area += 1
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                for next_x, next_y in (
                    (x - 1, y),
                    (x + 1, y),
                    (x, y - 1),
                    (x, y + 1),
                ):
                    if (
                        0 <= next_x < width
                        and 0 <= next_y < height
                        and mask[next_y, next_x]
                        and not visited[next_y, next_x]
                    ):
                        visited[next_y, next_x] = True
                        queue.append((next_x, next_y))

            boxes.append((min_x, min_y, max_x + 1, max_y + 1, area))

    return boxes


def _filter_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    *,
    image_width: int,
    image_height: int,
) -> list[tuple[int, int, int, int]]:
    page_area = image_width * image_height
    min_width = max(35, int(image_width * 0.06))
    min_height = max(35, int(image_height * 0.025))
    min_area = max(700, int(page_area * 0.002))
    max_area = int(page_area * 0.92)

    filtered: list[tuple[int, int, int, int]] = []
    for left, top, right, bottom, area in boxes:
        width = right - left
        height = bottom - top
        box_area = width * height

        if width < min_width or height < min_height:
            continue
        if area < min_area or box_area > max_area:
            continue
        filtered.append((left, top, right, bottom))

    return filtered


def _detect_article_boxes(image: Image.Image) -> list[tuple[int, int, int, int]]:
    detection_image, scale = _resize_for_detection(image)
    detect_width, detect_height = detection_image.size

    # Try larger dilation first for article-like blocks, then smaller dilation
    # if the page collapses into too few regions.
    candidate_boxes: list[tuple[int, int, int, int]] = []
    for dilation_ratio in (0.028, 0.018, 0.011):
        dilation_size = _make_odd(int(min(detect_width, detect_height) * dilation_ratio))
        mask = _content_mask(detection_image, dilation_size)
        boxes = _connected_component_boxes(mask)
        candidate_boxes = _filter_boxes(
            boxes,
            image_width=detect_width,
            image_height=detect_height,
        )
        if len(candidate_boxes) >= 2:
            break

    padding = max(10, int(min(image.size) * 0.01))
    scaled_boxes: list[tuple[int, int, int, int]] = []
    original_width, original_height = image.size

    for left, top, right, bottom in candidate_boxes:
        scaled_left = max(0, int(left / scale) - padding)
        scaled_top = max(0, int(top / scale) - padding)
        scaled_right = min(original_width, int(right / scale) + padding)
        scaled_bottom = min(original_height, int(bottom / scale) + padding)
        if scaled_right > scaled_left and scaled_bottom > scaled_top:
            scaled_boxes.append((scaled_left, scaled_top, scaled_right, scaled_bottom))

    return sorted(scaled_boxes, key=lambda box: (box[1], box[0]))


def _save_article_crops_from_pages(
    pages: list[Image.Image],
    output_dir: str | Path,
    *,
    max_crops_per_page: int = DEFAULT_MAX_CROPS_PER_PAGE,
) -> list[ArticleImageCrop]:
    destination = Path(output_dir)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    crops: list[ArticleImageCrop] = []
    for page_number, page_image in enumerate(pages, start=1):
        page = page_image.convert("RGB")
        boxes = _detect_article_boxes(page)[:max_crops_per_page]

        if not boxes:
            # Keep at least one visual cutout so users can inspect the scanned page.
            boxes = [(0, 0, page.width, page.height)]

        for page_crop_index, box in enumerate(boxes, start=1):
            crop_image = page.crop(box)
            crop_path = destination / f"page_{page_number:03d}_cutout_{page_crop_index:03d}.png"
            crop_image.save(crop_path, "PNG")
            crops.append(
                ArticleImageCrop(
                    index=len(crops) + 1,
                    page_number=page_number,
                    image_path=str(crop_path),
                    bbox=box,
                    width=crop_image.width,
                    height=crop_image.height,
                )
            )

    return crops


def crop_article_images_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    dpi: int = DEFAULT_DPI,
    max_crops_per_page: int = DEFAULT_MAX_CROPS_PER_PAGE,
) -> list[ArticleImageCrop]:
    """Convert a newspaper PDF to page images and save article-like cutouts."""
    pdf = Path(pdf_path)
    if not pdf.exists():
        raise ArticleCropError(f"PDF not found: {pdf}")

    try:
        from pdf2image import convert_from_path
    except ImportError as exc:
        raise ArticleCropError(
            "pdf2image is not installed. Install requirements.txt before cropping."
        ) from exc

    try:
        pages = convert_from_path(str(pdf), dpi=dpi, fmt="png", thread_count=2)
    except Exception as exc:
        raise ArticleCropError("Could not convert PDF pages into newspaper images.") from exc

    return _save_article_crops_from_pages(
        pages,
        output_dir,
        max_crops_per_page=max_crops_per_page,
    )


def crop_article_images_from_image(
    image_path: str | Path,
    output_dir: str | Path,
    *,
    max_crops_per_page: int = DEFAULT_MAX_CROPS_PER_PAGE,
) -> list[ArticleImageCrop]:
    """Save article-like cutouts from a newspaper JPG/PNG image."""
    image = Path(image_path)
    if not image.exists():
        raise ArticleCropError(f"Image not found: {image}")

    try:
        with Image.open(image) as opened:
            page = opened.convert("RGB")
    except Exception as exc:
        raise ArticleCropError("Could not open the uploaded newspaper image.") from exc

    return _save_article_crops_from_pages(
        [page],
        output_dir,
        max_crops_per_page=max_crops_per_page,
    )
