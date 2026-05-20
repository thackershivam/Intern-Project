"""Streamlit frontend for the Gujarati Newspaper Summarizer."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

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

LANGUAGE_OPTIONS = {
    "Gujarati": "gujarati",
    "Hindi": "hindi",
    "English": "english",
}
SPEED_OPTIONS = {
    "Normal": "normal",
    "Slow": "slow",
}
SUPPORTED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def save_uploaded_file(uploaded_file) -> Path:
    """Save a Streamlit UploadedFile into uploads/."""
    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower()
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("Please upload a valid PDF, PNG, JPG, or JPEG file.")

    upload_path = UPLOAD_DIR / f"{Path(original_name).stem}_{uuid4().hex[:8]}{extension}"
    upload_path.write_bytes(uploaded_file.getbuffer())
    return upload_path


def is_pdf(path: Path) -> bool:
    """Return True when the uploaded file is a PDF."""
    return path.suffix.lower() == ".pdf"


def extract_text_for_upload(upload_path: Path) -> tuple[str, str, int]:
    """Extract Gujarati text from either a PDF or newspaper image."""
    if is_pdf(upload_path):
        extraction = extract_text_from_pdf(upload_path)
        return extraction.text, extraction.method, extraction.page_count

    try:
        extracted_text = clean_text(extract_text_from_image(upload_path))
    except OCRProcessingError as exc:
        raise PDFExtractionError(str(exc)) from exc

    if not extracted_text:
        raise PDFExtractionError("No readable Gujarati text was found in this image.")

    return extracted_text, "paddleocr_image", 1


def crop_upload_images(upload_path: Path):
    """Create visual article cutouts from a PDF or newspaper image."""
    if is_pdf(upload_path):
        return crop_article_images_from_pdf(upload_path, CROP_DIR / upload_path.stem)
    return crop_article_images_from_image(upload_path, CROP_DIR / upload_path.stem)


def main() -> None:
    """Render the upload form and run the newspaper summarization pipeline."""
    st.set_page_config(
        page_title="Gujarati Newspaper Summarizer",
        page_icon=":newspaper:",
        layout="centered",
    )

    st.title("AI-powered Gujarati Newspaper Summarizer")
    st.write(
        "Upload one Gujarati newspaper PDF or newspaper image to cut article "
        "areas, summarize key news, and generate audio."
    )

    uploaded_file = st.file_uploader(
        "Upload one Gujarati newspaper PDF/image",
        type=["pdf", "png", "jpg", "jpeg"],
    )

    col1, col2 = st.columns(2)
    with col1:
        language_label = st.selectbox(
            "Summary / audio language",
            options=list(LANGUAGE_OPTIONS.keys()),
            index=0,
        )
    with col2:
        speed_label = st.selectbox(
            "Voice speed",
            options=list(SPEED_OPTIONS.keys()),
            index=0,
        )

    generate_summary = st.checkbox(
        "Generate AI summary and audio",
        value=True,
        help="Turn this off if you only want newspaper image cutouts.",
    )

    process_clicked = st.button(
        "Process File",
        type="primary",
        disabled=uploaded_file is None,
    )

    if not process_clicked:
        return

    if uploaded_file is None:
        st.warning("Please upload a PDF or image first.")
        return

    progress = st.progress(0, text="Starting processing...")

    try:
        with st.spinner("Saving uploaded file..."):
            upload_path = save_uploaded_file(uploaded_file)
            progress.progress(15, text="File saved in uploads/")

        crop_warning = None
        with st.spinner("Cutting article areas from newspaper photos..."):
            try:
                image_crops = crop_upload_images(upload_path)
            except ArticleCropError as exc:
                image_crops = []
                crop_warning = str(exc)
            progress.progress(45, text=f"Created {len(image_crops)} image cutouts")

        extracted_text = ""
        extraction_method = "not requested"
        page_count = 1
        articles = []
        summary = None
        audio_path = None

        if generate_summary:
            with st.spinner("Extracting Gujarati text..."):
                extracted_text, extraction_method, page_count = extract_text_for_upload(
                    upload_path
                )
                progress.progress(
                    60,
                    text=f"Text extracted using {extraction_method}",
                )

            with st.spinner("Separating articles by paragraph..."):
                articles = separate_articles_by_paragraph(extracted_text)
                summary_input = combine_articles_for_summary(articles, extracted_text)
                progress.progress(
                    68,
                    text=f"Separated {len(articles)} text sections",
                )

            with st.spinner("Generating AI summary with Gemini..."):
                summary = summarize_newspaper(
                    summary_input,
                    target_language=LANGUAGE_OPTIONS[language_label],
                )
                progress.progress(82, text="Summary generated")

            with st.spinner("Generating Gujarati audio..."):
                audio_path = generate_audio_summary(
                    summary.summary_text,
                    output_path=AUDIO_DIR / "summary.mp3",
                    language=LANGUAGE_OPTIONS[language_label],
                    speed=SPEED_OPTIONS[speed_label],
                )
                progress.progress(100, text="Done")
        else:
            progress.progress(
                100,
                text="Done. Summary and audio were skipped.",
            )

    except ValueError as exc:
        st.error(str(exc))
        return
    except PDFExtractionError as exc:
        st.error(f"Text extraction failed: {exc}")
        return
    except SummarizationError as exc:
        st.error(f"Gemini summarization failed: {exc}")
        return
    except TTSError as exc:
        st.error(f"Audio generation failed: {exc}")
        return
    except Exception as exc:  # Keep the UI user-friendly for unexpected failures.
        st.error(f"Unexpected error: {exc}")
        return

    st.success("Newspaper processed successfully!")
    st.caption(
        f"Input: {'PDF' if is_pdf(upload_path) else 'Image'} | "
        f"Pages: {page_count or 'Unknown'} | "
        f"Extraction: {extraction_method} | "
        f"Articles: {len(articles)} | "
        f"Image cutouts: {len(image_crops)} | "
        f"Gemini chunks: {summary.chunk_count if summary else 0}"
    )

    if crop_warning:
        st.warning(f"Image cutout warning: {crop_warning}")

    st.subheader("Newspaper Article Image Cutouts")
    if image_crops:
        st.write("These are cropped directly from the newspaper page photo/PDF image.")
        for crop in image_crops:
            with st.expander(
                f"Cutout {crop.index} - Page {crop.page_number}",
                expanded=crop.index == 1,
            ):
                st.caption(
                    f"Size: {crop.width} x {crop.height}px | "
                    f"Box: {crop.bbox}"
                )
                st.image(crop.image_path, use_container_width=True)
                crop_bytes = Path(crop.image_path).read_bytes()
                st.download_button(
                    label=f"Download cutout {crop.index}",
                    data=crop_bytes,
                    file_name=Path(crop.image_path).name,
                    mime="image/png",
                    key=f"download_crop_{crop.index}",
                )
    else:
        st.info("No image cutouts were created from this file.")

    if generate_summary:
        st.subheader("Text Sections")
        if articles:
            for article in articles:
                with st.expander(
                    f"Article {article.index}: {article.title}",
                    expanded=article.index == 1,
                ):
                    st.caption(
                        f"{article.word_count} words | {article.char_count} characters"
                    )
                    st.write(article.text)
        else:
            st.info("No separate article paragraphs were found.")

        if summary:
            st.subheader("Extracted Summary")
            st.markdown(summary.summary_text)

        if audio_path:
            st.subheader("Audio Summary")
            audio_bytes = audio_path.read_bytes()
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                label="Download audio summary",
                data=audio_bytes,
                file_name="summary.mp3",
                mime="audio/mpeg",
            )


if __name__ == "__main__":
    main()
