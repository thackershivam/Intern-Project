"""FastAPI backend for the AI-powered Gujarati Newspaper Summarizer."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from utils.articles import combine_articles_for_summary, separate_articles_by_paragraph
from utils.cleaner import clean_text
from utils.cropper import (
    ArticleCropError,
    crop_article_images_from_image,
    crop_article_images_from_pdf,
)
from utils.extractor import PDFExtractionError, extract_text_from_pdf
from utils.ocr import OCRProcessingError, extract_text_from_image
from utils.summarizer import SummarizationError, summarize_newspaper
from utils.tts import TTSError, generate_audio_summary


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
CROP_DIR = BASE_DIR / "article_crops"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {"gujarati", "hindi", "english"}
SUPPORTED_SPEEDS = {"normal", "slow"}
SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

app = FastAPI(
    title="Gujarati Newspaper Summarizer",
    description=(
        "Upload a Gujarati newspaper PDF or image and receive visual cutouts, "
        "a simple summary, and audio."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _save_upload(upload: UploadFile) -> Path:
    """Persist the uploaded PDF or newspaper image inside uploads/."""
    original_name = Path(upload.filename or "newspaper.pdf").name
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_INPUT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, PNG, JPG, and JPEG files are accepted.",
        )

    saved_path = UPLOAD_DIR / f"{Path(original_name).stem}_{uuid4().hex[:8]}{extension}"
    try:
        with saved_path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not save uploaded file.") from exc
    finally:
        upload.file.close()

    return saved_path


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def _extract_text_for_upload(path: Path) -> tuple[str, str, int]:
    """Extract text from either a PDF or direct newspaper image upload."""
    if _is_pdf(path):
        extraction = extract_text_from_pdf(path)
        return extraction.text, extraction.method, extraction.page_count

    try:
        image_text = clean_text(extract_text_from_image(path))
    except OCRProcessingError as exc:
        raise PDFExtractionError(str(exc)) from exc

    if not image_text:
        raise PDFExtractionError("No readable Gujarati text was found in this image.")

    return image_text, "paddleocr_image", 1


def _crop_upload_images(path: Path) -> tuple[list[object], str | None]:
    """Create visual article cutouts for PDFs or direct image uploads."""
    try:
        if _is_pdf(path):
            crops = crop_article_images_from_pdf(path, CROP_DIR / path.stem)
        else:
            crops = crop_article_images_from_image(path, CROP_DIR / path.stem)
        return crops, None
    except ArticleCropError as exc:
        return [], str(exc)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check for deployment and local testing."""
    return {"status": "ok"}


@app.post("/summarize")
def summarize_pdf(
    file: UploadFile = File(...),
    target_language: str = Form("gujarati"),
    voice_speed: str = Form("normal"),
    generate_summary: bool = Form(True),
) -> dict[str, object]:
    """Process an uploaded Gujarati newspaper PDF or image end-to-end."""
    normalized_language = target_language.lower().strip()
    normalized_speed = voice_speed.lower().strip()

    if normalized_language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language. Choose gujarati, hindi, or english.",
        )
    if normalized_speed not in SUPPORTED_SPEEDS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported voice speed. Choose normal or slow.",
        )

    uploaded_path = _save_upload(file)

    try:
        image_crops, crop_error = _crop_upload_images(uploaded_path)
        extracted_text = ""
        extraction_method = "not_requested"
        page_count = 1
        articles = []
        summary_text = None
        summary_language = normalized_language
        summary_model = None
        chunk_count = 0
        audio_path = None

        if generate_summary:
            extracted_text, extraction_method, page_count = _extract_text_for_upload(
                uploaded_path
            )
            articles = separate_articles_by_paragraph(extracted_text)
            summary_input = combine_articles_for_summary(articles, extracted_text)
            summary = summarize_newspaper(
                summary_input,
                target_language=normalized_language,
            )
            summary_text = summary.summary_text
            summary_language = summary.language
            summary_model = summary.model
            chunk_count = summary.chunk_count
            audio_path = generate_audio_summary(
                summary.summary_text,
                output_path=AUDIO_DIR / "summary.mp3",
                language=normalized_language,
                speed=normalized_speed,
            )
    except PDFExtractionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SummarizationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except TTSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "message": "File processed successfully.",
        "summary": summary_text,
        "language": summary_language,
        "model": summary_model,
        "chunk_count": chunk_count,
        "article_count": len(articles),
        "articles": [article.to_dict() for article in articles],
        "article_image_count": len(image_crops),
        "article_image_crops": [
            {
                "index": crop.index,
                "page_number": crop.page_number,
                "image_file": str(Path(crop.image_path).relative_to(BASE_DIR)),
                "image_url": (
                    f"/article-crops/{Path(crop.image_path).parent.name}/"
                    f"{Path(crop.image_path).name}"
                ),
                "bbox": crop.bbox,
                "width": crop.width,
                "height": crop.height,
            }
            for crop in image_crops
        ],
        "crop_error": crop_error,
        "extraction_method": extraction_method,
        "page_count": page_count,
        "uploaded_file": str(uploaded_path.relative_to(BASE_DIR)),
        "input_type": "pdf" if _is_pdf(uploaded_path) else "image",
        "audio_file": str(audio_path.relative_to(BASE_DIR)) if audio_path else None,
        "audio_url": f"/audio/{audio_path.name}" if audio_path else None,
    }


@app.get("/audio/{filename}")
def download_audio(filename: str) -> FileResponse:
    """Serve generated MP3 summaries."""
    audio_path = AUDIO_DIR / Path(filename).name
    if not audio_path.exists() or audio_path.suffix.lower() != ".mp3":
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(
        path=audio_path,
        media_type="audio/mpeg",
        filename=audio_path.name,
    )


@app.get("/article-crops/{run_id}/{filename}")
def download_article_crop(run_id: str, filename: str) -> FileResponse:
    """Serve cropped newspaper article images."""
    crop_path = CROP_DIR / Path(run_id).name / Path(filename).name
    if not crop_path.exists() or crop_path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Article crop not found.")

    return FileResponse(
        path=crop_path,
        media_type="image/png",
        filename=crop_path.name,
    )
