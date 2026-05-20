# Gujarati Newspaper Article Detection and Voice Reader

An AI-powered Streamlit application that works like a smart Gujarati newspaper
reader:

1. Upload a Gujarati newspaper image or PDF.
2. Detect individual article regions visually with OpenCV.
3. Display numbered article boundaries on the newspaper.
4. Select any article from preview cards.
5. Run PaddleOCR only on the selected article crop.
6. Convert the selected article text into Gujarati audio with gTTS.
7. Play or download the generated MP3.

## Features

- Upload support for:
  - JPEG
  - PNG
  - PDF first page
- OpenCV newspaper layout detection:
  - grayscale conversion
  - adaptive thresholding
  - morphological closing/dilation
  - contour detection
  - contour filtering
  - fallback column segmentation
- Numbered article boxes drawn over the newspaper image.
- Selectable article preview cards with thumbnails.
- OCR runs only on the selected article, not the full newspaper.
- Gujarati OCR using PaddleOCR.
- OCR preprocessing for selected crops.
- Gujarati MP3 generation using gTTS.
- Normal and slow audio speed options.
- Stop/clear audio control.
- Temporary upload cleanup after processing.
- Current-session crop/audio cleanup when clearing session or processing a new
  newspaper.

## Folder Structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── uploads/
│   └── .gitkeep
├── temp/
│   └── .gitkeep
├── audio/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    ├── layout_detector.py
    ├── ocr.py
    ├── tts.py
    ├── cleaner.py
    ├── pdf_handler.py
    └── visualization.py
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install Python packages:

```bash
pip install -r requirements.txt
```

## Poppler for PDF Support

PDF uploads require Poppler because `pdf2image` converts the first PDF page into
an image.

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
2. Extract it, for example to `C:\poppler`.
3. Add the Poppler `bin` folder to PATH, for example:

```text
C:\poppler\Library\bin
```

If Poppler is not in PATH, set an environment variable before running Streamlit:

```powershell
$env:POPPLER_PATH="C:\poppler\Library\bin"
```

## Run the App

```bash
streamlit run app.py
```

Or:

```bash
python -m streamlit run app.py
```

Open the URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Usage

1. Upload a Gujarati newspaper JPEG, PNG, or PDF.
2. Click **Detect Articles**.
3. View the newspaper with numbered article boxes.
4. Click an article preview button.
5. Click **Extract Text + Generate Audio**.
6. Read the extracted Gujarati text.
7. Play or download the generated audio.

## Processing Flow

```text
Upload Gujarati newspaper
→ Convert PDF first page if needed
→ Detect article regions with OpenCV
→ Crop every detected article
→ Draw numbered article boxes
→ User selects article
→ OCR selected article only with PaddleOCR
→ Clean Gujarati OCR text
→ Convert selected article to Gujarati speech
→ Play/download audio
```

## OpenCV Layout Detection

The detector in `utils/layout_detector.py` uses:

- RGB to grayscale conversion
- Gaussian blur
- adaptive thresholding
- morphological close and dilation with multiple kernels
- contour detection
- size, border, area, and fill-ratio filtering
- overlap removal
- fallback column segmentation if too few regions are found

The goal is to keep meaningful article-like regions and ignore tiny text pieces,
page borders, small icons, and decorative noise where possible.

## OCR Optimization

The app intentionally avoids OCR on the full newspaper page. OCR runs only when
the user selects an article crop. This keeps processing faster and reduces OCR
noise.

## Temporary File Cleanup

- Uploaded source files are deleted immediately after image conversion and
  layout detection.
- Cropped article images are stored under `temp/<session>/articles/` only for
  the current Streamlit session.
- Generated audio files are stored in `audio/article_<id>.mp3`.
- Clicking **Clear Session** or processing a new file removes previous
  temporary crops and audio.

## Notes

- PaddleOCR may download models the first time it runs.
- gTTS requires internet access.
- Very complex newspaper layouts, heavy ads, or low-resolution images may need
  manual tuning of detection thresholds.
- LayoutParser/Detectron2 is not required for this implementation, but can be
  added later for more advanced segmentation.
