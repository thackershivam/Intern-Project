"""Streamlit frontend for the Gujarati Newspaper Summarizer."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from utils.extractor import PDFExtractionError, extract_text_from_pdf
from utils.summarizer import SummarizationError, summarize_newspaper
from utils.tts import TTSError, generate_audio_summary


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

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
        "Upload a Gujarati newspaper PDF to extract text, summarize key news, "
        "and generate a Gujarati audio summary."
    )

    uploaded_pdf = st.file_uploader("Upload Gujarati newspaper PDF", type=["pdf"])

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
                45,
                text=f"Text extracted using {extraction.method}",
            )

        with st.spinner("Generating AI summary with Gemini..."):
            summary = summarize_newspaper(
                extraction.text,
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
        f"Gemini chunks: {summary.chunk_count}"
    )

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
