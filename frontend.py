"""Streamlit frontend for the Gujarati Newspaper Summarizer."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from utils.articles import combine_articles_for_summary, separate_articles_by_paragraph
from utils.cropper import ArticleCropError, crop_article_images_from_pdf
from utils.extractor import PDFExtractionError, extract_text_from_pdf
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


def save_uploaded_pdf(uploaded_file) -> Path:
    """Save a Streamlit UploadedFile into uploads/."""
    original_name = Path(uploaded_file.name).name
    if Path(original_name).suffix.lower() != ".pdf":
        raise ValueError("Please upload a valid PDF file.")

    pdf_path = UPLOAD_DIR / f"{Path(original_name).stem}_{uuid4().hex[:8]}.pdf"
    pdf_path.write_bytes(uploaded_file.getbuffer())
    return pdf_path


def main() -> None:
    """Render the upload form and run the newspaper summarization pipeline."""
    st.set_page_config(
        page_title="Gujarati Newspaper Summarizer",
        page_icon=":newspaper:",
        layout="centered",
    )

    st.title("AI-powered Gujarati Newspaper Summarizer")
    st.write(
        "Upload one Gujarati newspaper PDF to extract text, cut article areas "
        "from the newspaper page images, summarize key news, and generate audio."
    )

    uploaded_pdf = st.file_uploader("Upload one Gujarati newspaper PDF", type=["pdf"])

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

    process_clicked = st.button(
        "Process PDF",
        type="primary",
        disabled=uploaded_pdf is None,
    )

    if not process_clicked:
        return

    if uploaded_pdf is None:
        st.warning("Please upload a PDF first.")
        return

    progress = st.progress(0, text="Starting processing...")

    try:
        with st.spinner("Saving uploaded PDF..."):
            pdf_path = save_uploaded_pdf(uploaded_pdf)
            progress.progress(15, text="PDF saved in uploads/")

        with st.spinner("Extracting Gujarati text..."):
            extraction = extract_text_from_pdf(pdf_path)
            progress.progress(
                40,
                text=f"Text extracted using {extraction.method}",
            )

        with st.spinner("Separating articles by paragraph..."):
            articles = separate_articles_by_paragraph(extraction.text)
            summary_input = combine_articles_for_summary(articles, extraction.text)
            progress.progress(50, text=f"Separated {len(articles)} text sections")

        crop_warning = None
        with st.spinner("Cutting article areas from newspaper photos..."):
            try:
                image_crops = crop_article_images_from_pdf(
                    pdf_path,
                    CROP_DIR / pdf_path.stem,
                )
            except ArticleCropError as exc:
                image_crops = []
                crop_warning = str(exc)
            progress.progress(65, text=f"Created {len(image_crops)} image cutouts")

        with st.spinner("Generating AI summary with Gemini..."):
            summary = summarize_newspaper(
                summary_input,
                target_language=LANGUAGE_OPTIONS[language_label],
            )
            progress.progress(75, text="Summary generated")

        with st.spinner("Generating Gujarati audio..."):
            audio_path = generate_audio_summary(
                summary.summary_text,
                output_path=AUDIO_DIR / "summary.mp3",
                language=LANGUAGE_OPTIONS[language_label],
                speed=SPEED_OPTIONS[speed_label],
            )
            progress.progress(100, text="Done")

    except ValueError as exc:
        st.error(str(exc))
        return
    except PDFExtractionError as exc:
        st.error(f"PDF extraction failed: {exc}")
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
        f"Pages: {extraction.page_count or 'Unknown'} | "
        f"Extraction: {extraction.method} | "
        f"Articles: {len(articles)} | "
        f"Image cutouts: {len(image_crops)} | "
        f"Gemini chunks: {summary.chunk_count}"
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
        st.info("No image cutouts were created from this PDF.")

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

    st.subheader("Extracted Summary")
    st.markdown(summary.summary_text)

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
