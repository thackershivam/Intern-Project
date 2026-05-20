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
from utils.extractor import PDFExtractionError, extract_text_from_pdf
from utils.summarizer import SummarizationError, summarize_newspaper
from utils.tts import TTSError, generate_audio_summary


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

SUPPORTED_LANGUAGES = {"gujarati", "hindi", "english"}
SUPPORTED_SPEEDS = {"normal", "slow"}

app = FastAPI(
    title="Gujarati Newspaper Summarizer",
    description="Upload a Gujarati newspaper PDF and receive a simple summary with audio.",
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
    """Persist the uploaded PDF inside uploads/ with a safe unique name."""
    original_name = Path(upload.filename or "newspaper.pdf").name
    if Path(original_name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    saved_path = UPLOAD_DIR / f"{Path(original_name).stem}_{uuid4().hex[:8]}.pdf"
    try:
        with saved_path.open("wb") as destination:
            shutil.copyfileobj(upload.file, destination)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not save uploaded PDF.") from exc
    finally:
        upload.file.close()

    return saved_path


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check for deployment and local testing."""
    return {"status": "ok"}


@app.post("/summarize")
def summarize_pdf(
    file: UploadFile = File(...),
    target_language: str = Form("gujarati"),
    voice_speed: str = Form("normal"),
) -> dict[str, object]:
    """Process an uploaded Gujarati newspaper PDF end-to-end."""
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

    pdf_path = _save_upload(file)

    try:
        extraction = extract_text_from_pdf(pdf_path)
        articles = separate_articles_by_paragraph(extraction.text)
        summary_input = combine_articles_for_summary(articles, extraction.text)
        summary = summarize_newspaper(
            summary_input,
            target_language=normalized_language,
        )
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
        "message": "PDF processed successfully.",
        "summary": summary.summary_text,
        "language": summary.language,
        "model": summary.model,
        "chunk_count": summary.chunk_count,
        "article_count": len(articles),
        "articles": [article.to_dict() for article in articles],
        "extraction_method": extraction.method,
        "page_count": extraction.page_count,
        "uploaded_pdf": str(pdf_path.relative_to(BASE_DIR)),
        "audio_file": str(audio_path.relative_to(BASE_DIR)),
        "audio_url": f"/audio/{audio_path.name}",
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
