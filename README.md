# AI-powered Gujarati Newspaper Summarizer

This project lets users upload one Gujarati newspaper PDF, extract Gujarati
text, cut article areas out of the newspaper page images, summarize the
important news with Gemini, and generate a Gujarati MP3 audio summary with gTTS.

## Features

- Streamlit frontend with one-PDF upload, progress bar, newspaper image
  cutouts, separated text sections, summary display, audio player, and MP3
  download.
- FastAPI backend for API-based PDF processing.
- Text extraction with `pdfplumber` first for selectable PDFs.
- Automatic OCR fallback with `pdf2image` and `PaddleOCR` for scanned PDFs.
- Gujarati text cleaning to reduce OCR noise and duplicate lines.
- Newspaper photo/PDF image cutouts saved as PNG files in `article_crops/`.
- Paragraph-based article separation with article count, title, text, word
  count, and character count.
- Gemini-based chunked summarization for large newspaper PDFs.
- Output includes:
  - A short summary
  - 5 important news bullet points
  - Likely news categories
- gTTS audio generation in Gujarati, Hindi, or English.
- Voice speed option: normal or slow.

## Project Structure

```text
project/
├── app.py
├── frontend.py
├── requirements.txt
├── .env.example
├── uploads/
│   └── .gitkeep
├── audio/
│   └── .gitkeep
├── article_crops/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    ├── articles.py
    ├── cropper.py
    ├── extractor.py
    ├── ocr.py
    ├── summarizer.py
    ├── tts.py
    └── cleaner.py
```

## Requirements

- Python 3.10+
- Gemini API key
- Poppler for PDF-to-image conversion
- Internet access for Gemini and gTTS

### Install Poppler

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

macOS:

```bash
brew install poppler
```

Windows:

Install Poppler for Windows and add the `bin` directory to your `PATH`.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file:

```bash
cp .env.example .env
```

4. Add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
```

## Run the Backend

```bash
uvicorn app:app --reload
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

### Backend Endpoints

- `GET /health` - health check
- `POST /summarize` - upload and process one PDF
- `GET /audio/{filename}` - download generated audio
- `GET /article-crops/{run_id}/{filename}` - download generated image cutouts

Example API request:

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
  -F "file=@newspaper.pdf" \
  -F "target_language=gujarati" \
  -F "voice_speed=normal"
```

## Run the Frontend

```bash
streamlit run frontend.py
```

Then open the Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. The user uploads one PDF from the Streamlit UI or FastAPI endpoint.
2. The file is saved in `uploads/`.
3. `utils.extractor` validates the PDF and tries `pdfplumber`.
4. If extracted text is empty or too small, `utils.ocr` converts PDF pages to
   images with `pdf2image` and runs PaddleOCR with Gujarati language support.
5. `utils.cleaner` removes noisy characters, duplicate lines, and unnecessary
   spaces while preserving paragraph breaks.
6. `utils.articles` separates article sections by paragraph and labels each
   section with article number, title, word count, and character count.
7. `utils.cropper` converts the PDF into page images and cuts article-like
   visual blocks from the newspaper photo/scanned page. These PNG files are
   saved in `article_crops/`.
8. `utils.summarizer` sends labeled article sections to Gemini, splitting long
   text into chunks when needed.
9. Gemini returns a simple-language summary, 5 bullet points, and categories.
10. `utils.tts` converts the final summary into MP3 audio with gTTS.
11. The generated audio is saved in `audio/summary.mp3` and can be played or
   downloaded.

## Newspaper Image Cutouts

The app now creates visual cutouts directly from the newspaper PDF/page photo.
It converts the PDF pages to images, detects dark content blocks such as article
text areas and photos, and saves each detected block as a PNG.

In Streamlit, these appear under:

```text
Newspaper Article Image Cutouts
```

Each cutout can be viewed and downloaded separately.

The FastAPI `/summarize` response includes:

- `article_image_count`
- `article_image_crops`
  - `index`
  - `page_number`
  - `image_file`
  - `image_url`
  - `bbox`
  - `width`
  - `height`
- `crop_error`

## Paragraph-Based Article Separation

The app still also uses paragraph boundaries from the extracted PDF text to
create text sections for summarization. Short paragraphs are treated as possible
headlines and attached to the next larger paragraph. If OCR or PDF extraction
does not preserve blank lines, the app falls back to separating visible text
lines so users can still inspect the PDF content in sections.

The FastAPI `/summarize` response includes:

- `article_count`
- `articles`
  - `index`
  - `title`
  - `text`
  - `word_count`
  - `char_count`

## Error Handling

The app handles common failures:

- Invalid or non-PDF uploads
- Corrupt PDFs
- Empty extracted text
- OCR initialization or processing failures
- Missing `GEMINI_API_KEY`
- Gemini API errors
- gTTS audio generation errors
- PDF-to-image cutout errors

Errors are returned as HTTP errors in FastAPI and displayed as Streamlit
messages in the frontend.

## Notes

- `uploads/`, `audio/`, and `article_crops/` are runtime folders. Their
  generated files are ignored by Git.
- `audio/summary.mp3` is overwritten each time a new summary is generated.
- PaddleOCR model downloads may happen on the first OCR run.
- For best OCR results, use clear newspaper scans with readable Gujarati text.
