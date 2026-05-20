"""Streamlit app for Gujarati newspaper article detection and voice reading."""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

import streamlit as st
from PIL import Image

from utils.layout_detector import ArticleRegion, LayoutDetectionError, detect_articles
from utils.ocr import OCRProcessingError, extract_gujarati_text_from_article
from utils.pdf_handler import InputFileError, is_supported_file, load_newspaper_image
from utils.tts import TTSError, generate_gujarati_audio
from utils.visualization import create_thumbnail, draw_article_boxes


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
AUDIO_DIR = BASE_DIR / "audio"

for directory in (UPLOAD_DIR, TEMP_DIR, AUDIO_DIR):
    directory.mkdir(exist_ok=True)


def cleanup_directory(path: Path) -> None:
    """Remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def cleanup_audio_files() -> None:
    """Delete generated audio files from older selections."""
    for audio_file in AUDIO_DIR.glob("article_*.mp3"):
        audio_file.unlink(missing_ok=True)


def reset_processing_state() -> None:
    """Clear previous Streamlit state and temporary files before new processing."""
    previous_session_dir = st.session_state.get("session_dir")
    if previous_session_dir:
        cleanup_directory(Path(previous_session_dir))
    cleanup_audio_files()

    for key in (
        "session_id",
        "session_dir",
        "newspaper_image",
        "regions",
        "selected_article_id",
        "ocr_text",
        "audio_path",
        "uploaded_name",
    ):
        st.session_state.pop(key, None)


def save_uploaded_file(uploaded_file, session_dir: Path) -> Path:
    """Save an uploaded PDF/image temporarily for conversion/detection."""
    original_name = Path(uploaded_file.name).name
    if not is_supported_file(original_name):
        raise InputFileError("Please upload a JPEG, PNG, or PDF newspaper file.")

    upload_path = session_dir / "upload" / original_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(uploaded_file.getbuffer())
    return upload_path


def process_uploaded_file(uploaded_file) -> None:
    """Load the upload, detect article regions, and store results in session state."""
    reset_processing_state()
    session_id = uuid4().hex[:10]
    session_dir = TEMP_DIR / session_id
    crop_dir = session_dir / "articles"
    session_dir.mkdir(parents=True, exist_ok=True)

    upload_path: Path | None = None
    try:
        upload_path = save_uploaded_file(uploaded_file, session_dir)
        image = load_newspaper_image(upload_path)
        regions = detect_articles(image, crop_dir)
    finally:
        # The uploaded source file is not needed after conversion/detection.
        if upload_path and upload_path.exists():
            upload_path.unlink(missing_ok=True)

    st.session_state.session_id = session_id
    st.session_state.session_dir = str(session_dir)
    st.session_state.newspaper_image = image
    st.session_state.regions = regions
    st.session_state.selected_article_id = regions[0].article_id if regions else None
    st.session_state.ocr_text = None
    st.session_state.audio_path = None
    st.session_state.uploaded_name = uploaded_file.name


def get_selected_region() -> ArticleRegion | None:
    """Return the currently selected article region."""
    selected_id = st.session_state.get("selected_article_id")
    regions: list[ArticleRegion] = st.session_state.get("regions", [])
    return next((region for region in regions if region.article_id == selected_id), None)


def select_article(article_id: int) -> None:
    """Select an article and clear OCR/audio from any previous selection."""
    st.session_state.selected_article_id = article_id
    st.session_state.ocr_text = None
    st.session_state.audio_path = None
    cleanup_audio_files()


def render_article_cards(regions: list[ArticleRegion]) -> None:
    """Render selectable article preview cards."""
    st.subheader("Select an Article")
    st.caption("Click an article button below. OCR runs only after you select an article.")

    columns_per_row = 3
    for start in range(0, len(regions), columns_per_row):
        columns = st.columns(columns_per_row)
        for column, region in zip(columns, regions[start : start + columns_per_row]):
            with column:
                selected = region.article_id == st.session_state.get("selected_article_id")
                border_color = "#00b050" if selected else "#dddddd"
                st.markdown(
                    f"""
                    <div class="article-card" style="border-color:{border_color};">
                        <strong>Article {region.article_id}</strong><br/>
                        <small>{region.width} x {region.height}px</small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(create_thumbnail(region.crop_path), use_container_width=True)
                if st.button(
                    f"{'Selected' if selected else 'Select'} Article {region.article_id}",
                    key=f"select_{region.article_id}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    select_article(region.article_id)
                    st.rerun()


def render_selected_article(region: ArticleRegion) -> None:
    """Render selected article crop and OCR/audio controls."""
    st.subheader(f"Selected Article {region.article_id}")
    st.image(region.crop_path, caption="Selected article crop", use_container_width=True)

    speed = st.radio(
        "Gujarati voice speed",
        options=["normal", "slow"],
        horizontal=True,
        help="gTTS supports normal and slow Gujarati speech.",
    )

    col1, col2 = st.columns(2)
    with col1:
        extract_clicked = st.button(
            "Extract Text + Generate Audio",
            type="primary",
            use_container_width=True,
        )
    with col2:
        if st.button("Stop / Clear Audio", use_container_width=True):
            st.session_state.audio_path = None
            cleanup_audio_files()
            st.info("Audio cleared. Browser playback can also be paused from the audio player.")

    if extract_clicked:
        with st.spinner("Running PaddleOCR only on the selected article..."):
            text = extract_gujarati_text_from_article(region.crop_path)
            st.session_state.ocr_text = text

        with st.spinner("Generating Gujarati voice audio..."):
            audio_path = AUDIO_DIR / f"article_{region.article_id}.mp3"
            generate_gujarati_audio(st.session_state.ocr_text, audio_path, speed=speed)
            st.session_state.audio_path = str(audio_path)

    if st.session_state.get("ocr_text"):
        st.subheader("Extracted Gujarati Text")
        st.text_area(
            "OCR text",
            value=st.session_state.ocr_text,
            height=260,
            label_visibility="collapsed",
        )

    audio_path_value = st.session_state.get("audio_path")
    if audio_path_value and Path(audio_path_value).exists():
        audio_path = Path(audio_path_value)
        audio_bytes = audio_path.read_bytes()
        st.subheader("Gujarati Audio Reader")
        st.audio(audio_bytes, format="audio/mp3")
        st.download_button(
            "Download audio",
            data=audio_bytes,
            file_name=audio_path.name,
            mime="audio/mpeg",
            use_container_width=True,
        )


def main() -> None:
    """Render the Streamlit newspaper reader UI."""
    st.set_page_config(
        page_title="Gujarati Newspaper Reader",
        page_icon=":newspaper:",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .article-card {
            border: 2px solid #dddddd;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
            background: #fafafa;
            transition: all 0.15s ease-in-out;
        }
        .article-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Gujarati Newspaper Article Detection and Voice Reader")
    st.write(
        "Upload a Gujarati newspaper image or PDF. The app detects visual article "
        "regions first, then OCR and audio are generated only for the article you select."
    )

    with st.sidebar:
        st.header("Upload")
        uploaded_file = st.file_uploader(
            "Newspaper JPEG, PNG, or PDF",
            type=["jpg", "jpeg", "png", "pdf"],
        )
        process_clicked = st.button(
            "Detect Articles",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

        if st.button("Clear Session", use_container_width=True):
            reset_processing_state()
            st.rerun()

        st.divider()
        st.caption(
            "PDF support converts only the first page for fast detection. "
            "Set POPPLER_PATH if Poppler is not available in PATH."
        )

    if process_clicked and uploaded_file is not None:
        try:
            progress = st.progress(0, text="Saving upload...")
            with st.spinner("Detecting article layout with OpenCV..."):
                progress.progress(20, text="Loading newspaper image...")
                process_uploaded_file(uploaded_file)
                progress.progress(100, text="Article detection complete.")
            st.success(f"Detected {len(st.session_state.regions)} article regions.")
        except (InputFileError, LayoutDetectionError) as exc:
            reset_processing_state()
            st.error(str(exc))
        except Exception as exc:
            reset_processing_state()
            st.error(f"Unexpected processing error: {exc}")

    regions: list[ArticleRegion] = st.session_state.get("regions", [])
    newspaper_image: Image.Image | None = st.session_state.get("newspaper_image")

    if not regions or newspaper_image is None:
        st.info("Upload a newspaper file and click Detect Articles to begin.")
        return

    selected_region = get_selected_region()

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Detected Newspaper Layout")
        st.caption(
            "Numbered boxes show detected article regions. Select the matching article card to read it."
        )
        boxed_image = draw_article_boxes(
            newspaper_image,
            regions,
            selected_article_id=selected_region.article_id if selected_region else None,
        )
        st.image(boxed_image, use_container_width=True)

    with right:
        render_article_cards(regions)

    st.divider()

    if selected_region:
        try:
            render_selected_article(selected_region)
        except OCRProcessingError as exc:
            st.error(f"OCR failed: {exc}")
        except TTSError as exc:
            st.error(f"Audio generation failed: {exc}")


if __name__ == "__main__":
    main()
