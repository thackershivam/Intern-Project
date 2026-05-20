"""Streamlit app for column-wise newspaper OCR and audio reading."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image

from utils.cleaner import merge_column_texts
from utils.column_detector import ColumnDetectionError, ColumnRegion, detect_columns
from utils.ocr import OCRProcessingError, extract_text_column_wise
from utils.tts import TTSError, generate_audio
from utils.visualization import draw_column_boxes, make_thumbnail


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
AUDIO_DIR = BASE_DIR / "audio"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}
LANGUAGES = ["Gujarati", "Hindi", "English"]

for directory in (UPLOAD_DIR, TEMP_DIR, AUDIO_DIR):
    directory.mkdir(exist_ok=True)


class UploadError(RuntimeError):
    """Raised for invalid or unreadable uploads."""


def cleanup_old_audio() -> None:
    """Delete old generated audio before a new run."""
    for audio_file in AUDIO_DIR.glob("*.mp3"):
        audio_file.unlink(missing_ok=True)


def cleanup_temp_dir(path: Path | None) -> None:
    """Remove a temporary processing directory."""
    if path and path.exists():
        shutil.rmtree(path, ignore_errors=True)


def reset_results() -> None:
    """Clear previous UI results."""
    for key in (
        "processed_image",
        "boxed_image",
        "columns",
        "column_previews",
        "column_texts",
        "extracted_text",
        "audio_path",
        "source_name",
    ):
        st.session_state.pop(key, None)


def save_upload(uploaded_file, temp_dir: Path) -> Path:
    """Save upload temporarily for image/PDF loading."""
    filename = Path(uploaded_file.name).name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UploadError("Please upload a JPEG, PNG, or PDF file.")

    upload_path = temp_dir / "upload" / filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(uploaded_file.getbuffer())
    return upload_path


def load_upload_as_image(upload_path: Path) -> Image.Image:
    """Load an image upload or convert the first page of a PDF."""
    if upload_path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise UploadError("pdf2image is not installed. Install requirements.txt.") from exc

        poppler_path = os.getenv("POPPLER_PATH") or None
        try:
            pages = convert_from_path(
                str(upload_path),
                dpi=180,
                first_page=1,
                last_page=1,
                fmt="png",
                poppler_path=poppler_path,
            )
        except Exception as exc:
            raise UploadError(
                "Could not convert PDF. Install Poppler or set POPPLER_PATH."
            ) from exc
        if not pages:
            raise UploadError("The PDF does not contain any pages.")
        return pages[0].convert("RGB")

    try:
        with Image.open(upload_path) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise UploadError("Could not open the uploaded image.") from exc


def process_newspaper(uploaded_file, language: str, speed: str) -> None:
    """Run column detection, OCR, text merge, and audio generation."""
    reset_results()
    cleanup_old_audio()

    temp_dir = TEMP_DIR / uuid4().hex[:10]
    upload_path: Path | None = None
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        upload_path = save_upload(uploaded_file, temp_dir)
        image = load_upload_as_image(upload_path)

        progress = st.progress(0, text="Detecting newspaper columns...")
        processed_image, columns = detect_columns(image, temp_dir / "columns")
        progress.progress(30, text=f"Detected {len(columns)} columns")

        boxed_image = draw_column_boxes(processed_image, columns)
        previews = [make_thumbnail(column.crop_path) for column in columns]

        def update_ocr_progress(index: int, total: int, column: ColumnRegion) -> None:
            percent = 30 + int((index - 1) / max(total, 1) * 45)
            progress.progress(percent, text=f"OCR column {index} of {total}...")

        column_texts = extract_text_column_wise(
            columns,
            language,
            progress_callback=update_ocr_progress,
        )
        progress.progress(78, text="Merging column text in reading order...")
        extracted_text = merge_column_texts(column_texts)
        if not extracted_text:
            raise OCRProcessingError("No text was extracted from the detected columns.")

        progress.progress(88, text="Generating audio...")
        audio_path = generate_audio(
            extracted_text,
            language,
            AUDIO_DIR / "output.mp3",
            speed=speed,
        )
        progress.progress(100, text="Done")

        st.session_state.processed_image = processed_image
        st.session_state.boxed_image = boxed_image
        st.session_state.columns = columns
        st.session_state.column_previews = previews
        st.session_state.column_texts = column_texts
        st.session_state.extracted_text = extracted_text
        st.session_state.audio_path = str(audio_path)
        st.session_state.source_name = uploaded_file.name
    finally:
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)
        cleanup_temp_dir(temp_dir)


def render_column_previews() -> None:
    """Show detected column thumbnails in left-to-right order."""
    columns: list[ColumnRegion] = st.session_state.get("columns", [])
    previews: list[Image.Image] = st.session_state.get("column_previews", [])
    column_texts: list[str] = st.session_state.get("column_texts", [])

    if not columns:
        return

    st.subheader("Detected Columns")
    st.caption("OCR was performed column-wise: top-to-bottom inside each column, then left-to-right.")

    preview_columns = st.columns(min(4, len(columns)))
    for index, (column, preview) in enumerate(zip(columns, previews), start=1):
        with preview_columns[(index - 1) % len(preview_columns)]:
            st.markdown(f"**Column {column.column_id}**")
            st.image(preview, use_container_width=True)
            with st.expander(f"Text from column {column.column_id}"):
                text = column_texts[index - 1] if index - 1 < len(column_texts) else ""
                st.text(text or "No text detected.")


def render_results() -> None:
    """Render OCR text, visualization, and generated audio."""
    boxed_image = st.session_state.get("boxed_image")
    extracted_text = st.session_state.get("extracted_text")
    audio_path_value = st.session_state.get("audio_path")

    if boxed_image is None or not extracted_text:
        st.info("Upload a newspaper image/PDF and click Process Newspaper.")
        return

    st.success("Newspaper processed successfully.")
    st.subheader("Detected Column Layout")
    st.image(boxed_image, use_container_width=True)
    render_column_previews()

    st.subheader("Extracted Text")
    st.text_area(
        "Column-wise OCR output",
        value=extracted_text,
        height=360,
        label_visibility="collapsed",
    )

    if audio_path_value and Path(audio_path_value).exists():
        audio_path = Path(audio_path_value)
        audio_bytes = audio_path.read_bytes()
        st.subheader("Generated Audio")
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "Download audio",
            data=audio_bytes,
            file_name="output.mp3",
            mime="audio/mpeg",
            use_container_width=True,
        )


def main() -> None:
    """Render the Streamlit UI."""
    st.set_page_config(
        page_title="Newspaper Article Voice Reader",
        page_icon=":newspaper:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .main .block-container { padding-top: 2rem; }
        div[data-testid="stFileUploader"] section {
            border-radius: 14px;
            border: 1px dashed #7aa7ff;
            background: #f7faff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI-powered Newspaper Article Voice Reader")
    st.write(
        "Upload a Gujarati, Hindi, or English newspaper article image. "
        "The app detects columns first and OCRs each column in proper reading order."
    )

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader(
            "Upload newspaper image/PDF",
            type=["jpg", "jpeg", "png", "pdf"],
        )
        language = st.selectbox("Input / output language", LANGUAGES)
        speed = st.radio("Audio speed", ["normal", "slow"], horizontal=True)

        process_clicked = st.button(
            "Process Newspaper",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

        if st.button("Clear Results", use_container_width=True):
            reset_results()
            cleanup_old_audio()
            st.rerun()

        st.divider()
        st.caption(
            "Reading order is fixed as: Column 1 top-to-bottom, then Column 2, "
            "then Column 3/4 left-to-right."
        )

    if process_clicked and uploaded_file is not None:
        try:
            with st.spinner("Processing newspaper..."):
                process_newspaper(uploaded_file, language, speed)
        except (UploadError, ColumnDetectionError, OCRProcessingError, TTSError) as exc:
            reset_results()
            st.error(str(exc))
        except Exception as exc:
            reset_results()
            st.error(f"Unexpected error: {exc}")

    render_results()


if __name__ == "__main__":
    main()
