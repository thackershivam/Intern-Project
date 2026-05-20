"""FastAPI backend for the interactive newspaper article highlighter."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.utils.detector import (
    ArticleBox,
    DetectionError,
    detect_articles,
    load_upload_image,
    validate_extension,
)
from backend.utils.ocr import OCRError, extract_article_text
from backend.utils.tts import TTSError, generate_speech


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
AUDIO_DIR = BASE_DIR / "audio"

for directory in (UPLOAD_DIR, TEMP_DIR, AUDIO_DIR):
    directory.mkdir(exist_ok=True)

app = FastAPI(
    title="Interactive Newspaper Article Highlighter API",
    description="Detect newspaper articles, OCR selected articles, and generate optional audio.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session index. For production multi-worker deployments, move this
# to Redis/object storage.
SESSIONS: dict[str, dict[str, Any]] = {}
SESSION_TTL_SECONDS = 60 * 60


def _session_dir(session_id: str) -> Path:
    return TEMP_DIR / session_id


def _cleanup_session_files(session_id: str) -> None:
    """Remove temporary upload, processed image, article crops, and audio."""
    shutil.rmtree(_session_dir(session_id), ignore_errors=True)
    session_audio = AUDIO_DIR / session_id
    shutil.rmtree(session_audio, ignore_errors=True)
    SESSIONS.pop(session_id, None)


def _get_session(session_id: str) -> dict[str, Any]:
    session = SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired.")
    return session


def _cleanup_stale_sessions() -> None:
    """Best-effort cleanup for old temporary sessions."""
    now = time.time()
    stale_ids = [
        session_id
        for session_id, session in SESSIONS.items()
        if now - float(session.get("created_at", now)) > SESSION_TTL_SECONDS
    ]
    for session_id in stale_ids:
        _cleanup_session_files(session_id)


def _get_article(session: dict[str, Any], article_id: int) -> ArticleBox:
    for article in session["articles"]:
        if article.id == article_id:
            return article
    raise HTTPException(status_code=404, detail="Article not found.")


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/upload")
async def upload_newspaper(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a newspaper image/PDF and detect article boundaries.

    The original upload is deleted after processing; a resized processed image
    and article crops remain in a temporary session folder for interaction.
    """
    _cleanup_stale_sessions()
    original_name = Path(file.filename or "newspaper.png").name
    try:
        validate_extension(original_name)
    except DetectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    session_id = uuid4().hex[:12]
    session_dir = _session_dir(session_id)
    upload_dir = session_dir / "upload"
    crop_dir = session_dir / "crops"
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / original_name

    try:
        with upload_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        image = load_upload_image(upload_path)
        processed_image, articles = detect_articles(image, crop_dir)
        processed_path = session_dir / "newspaper.png"
        processed_image.save(processed_path)
    except DetectionError as exc:
        _cleanup_session_files(session_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _cleanup_session_files(session_id)
        raise HTTPException(status_code=500, detail="Failed to process upload.") from exc
    finally:
        file.file.close()
        upload_path.unlink(missing_ok=True)

    SESSIONS[session_id] = {
        "processed_path": str(processed_path),
        "articles": articles,
        "last_text": {},
        "created_at": time.time(),
    }

    return {
        "session_id": session_id,
        "image_url": f"/image/{session_id}",
        "width": processed_image.width,
        "height": processed_image.height,
        "articles": [article.public_dict() for article in articles],
    }


@app.get("/articles")
def get_articles(session_id: str) -> dict[str, Any]:
    """Return detected article boxes for a session."""
    session = _get_session(session_id)
    return {
        "session_id": session_id,
        "articles": [article.public_dict() for article in session["articles"]],
    }


@app.get("/image/{session_id}")
def get_processed_image(session_id: str) -> FileResponse:
    """Serve the resized newspaper image used by the canvas."""
    session = _get_session(session_id)
    image_path = Path(session["processed_path"])
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Processed image not found.")
    return FileResponse(image_path, media_type="image/png", filename="newspaper.png")


@app.post("/ocr")
def ocr_article(
    session_id: str = Form(...),
    article_id: int = Form(...),
    language: str = Form("Gujarati"),
) -> dict[str, Any]:
    """OCR only the clicked/selected article crop."""
    session = _get_session(session_id)
    article = _get_article(session, article_id)
    try:
        text = extract_article_text(article.crop_path, language)
    except OCRError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session["last_text"][article_id] = text
    return {
        "session_id": session_id,
        "article_id": article_id,
        "language": language,
        "text": text,
    }


@app.post("/tts")
def tts_article(
    session_id: str = Form(...),
    article_id: int = Form(...),
    language: str = Form("Gujarati"),
    text: str = Form(...),
) -> dict[str, str]:
    """Generate optional audio for extracted article text."""
    _get_session(session_id)
    output_dir = AUDIO_DIR / session_id
    output_path = output_dir / f"article_{article_id}.mp3"
    try:
        audio_path = generate_speech(text, language, output_path)
    except TTSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "audio_url": f"/audio/{session_id}/{audio_path.name}",
        "filename": audio_path.name,
    }


@app.get("/audio/{session_id}/{filename}")
def get_audio(session_id: str, filename: str) -> FileResponse:
    """Serve generated MP3 audio."""
    _get_session(session_id)
    audio_path = AUDIO_DIR / Path(session_id).name / Path(filename).name
    if not audio_path.exists() or audio_path.suffix.lower() != ".mp3":
        raise HTTPException(status_code=404, detail="Audio not found.")
    return FileResponse(audio_path, media_type="audio/mpeg", filename=audio_path.name)


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Explicitly delete temporary files for a session."""
    _cleanup_session_files(session_id)
    return {"status": "deleted"}
