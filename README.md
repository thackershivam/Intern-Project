# AI-powered Newspaper Article Voice Reader

This Streamlit application reads a newspaper article image in Gujarati, Hindi, or
English. It detects newspaper columns first, OCRs each column separately with
PaddleOCR, merges the text in correct reading order, and converts the extracted
text into speech with gTTS.

## Core Reading Logic

The app does **not** OCR the whole page randomly.

It follows this order:

```text
Column 1: top to bottom
Column 2: top to bottom
Column 3: top to bottom
Column 4: top to bottom
```

Columns are sorted:

```text
LEFT -> RIGHT
```

This is the key feature for newspaper layouts with 2, 3, or 4 columns.

## Features

- Upload support:
  - JPEG
  - PNG
  - Optional PDF first page
- Language dropdown:
  - Gujarati
  - Hindi
  - English
- PaddleOCR language mapping:
  - Gujarati -> `gu`
  - Hindi -> `hi`
  - English -> `en`
- OpenCV column detection:
  - grayscale conversion
  - adaptive thresholding
  - morphology
  - vertical whitespace detection
  - column segmentation
  - left-to-right sorting
- OCR each detected column separately.
- Merge extracted text in correct newspaper reading order.
- Show detected columns visually.
- Show extracted text.
- Generate audio:
  - Gujarati -> `gu`
  - Hindi -> `hi`
  - English -> `en`
- Audio player and MP3 download.
- Normal/slow speech speed option.
- Temporary upload and crop cleanup.

## Folder Structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── uploads/
│   └── .gitkeep
├── audio/
│   └── .gitkeep
├── temp/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    ├── column_detector.py
    ├── ocr.py
    ├── tts.py
    ├── cleaner.py
    └── visualization.py
```

## Installation

Create and activate a virtual environment.

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## PDF Support

PDF support is optional and uses `pdf2image`.

For PDFs, install Poppler.

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

1. Download Poppler for Windows.
2. Extract it, for example:

```text
C:\poppler
```

3. Add this folder to PATH:

```text
C:\poppler\Library\bin
```

Or set `POPPLER_PATH` before running:

```powershell
$env:POPPLER_PATH="C:\poppler\Library\bin"
```

## Run

```bash
streamlit run app.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## Usage

1. Upload a newspaper article image or PDF.
2. Select language:
   - Gujarati
   - Hindi
   - English
3. Choose audio speed:
   - normal
   - slow
4. Click **Process Newspaper**.
5. View detected columns.
6. Read extracted text.
7. Play or download `audio/output.mp3`.

## Processing Pipeline

```text
Upload image/PDF
-> Load image / convert first PDF page
-> Resize very large image
-> Grayscale
-> Adaptive threshold
-> Morphological cleanup
-> Detect vertical whitespace gutters
-> Split into columns
-> Sort columns left-to-right
-> OCR each column separately
-> Sort OCR lines top-to-bottom inside each column
-> Merge column text
-> Generate audio
```

## Temporary File Handling

- Uploaded files are saved temporarily under `temp/`.
- Cropped column images are saved temporarily under `temp/`.
- Temporary uploads and crops are deleted after processing.
- Old audio files are deleted before a new process.
- Final audio is saved as:

```text
audio/output.mp3
```

## Notes

- PaddleOCR may download language models the first time it runs.
- gTTS requires internet access.
- Very noisy scans may need better image quality for accurate OCR.
- The app is optimized to avoid OCR on tiny random regions and to preserve
  column-wise newspaper reading order.
