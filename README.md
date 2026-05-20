# Interactive Newspaper Article Highlighter

A full-stack AI-powered smart newspaper reader. Users upload a newspaper image
or PDF, the backend detects article regions, and the React frontend renders the
newspaper on an HTML5 canvas with real-time hover highlighting. Clicking an
article runs OCR only on that selected crop and can optionally generate voice
audio.

## Features

- Upload JPEG, PNG, or PDF.
- FastAPI backend.
- React + Tailwind CSS frontend.
- HTML5 Canvas newspaper viewer.
- Real-time mouse hover article highlighting.
- Click article to select it.
- PaddleOCR only on selected article.
- OCR language dropdown:
  - Gujarati -> `gu`
  - Hindi -> `hi`
  - English -> `en`
- Top-to-bottom, left-to-right OCR line ordering within selected article.
- Optional gTTS MP3 generation.
- Audio playback and download.
- OpenCV article detection:
  - grayscale conversion
  - adaptive thresholding
  - morphology
  - contour detection
  - contour filtering
  - fallback column segmentation
- Temporary file cleanup:
  - uploaded source file deleted after processing
  - session files deleted on frontend cleanup or API delete
  - stale session cleanup on new uploads

## Folder Structure

```text
project/
├── requirements.txt
├── README.md
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── uploads/
│   │   └── .gitkeep
│   ├── temp/
│   │   └── .gitkeep
│   ├── audio/
│   │   └── .gitkeep
│   └── utils/
│       ├── __init__.py
│       ├── detector.py
│       ├── ocr.py
│       ├── tts.py
│       └── cleaner.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── postcss.config.js
    ├── tailwind.config.js
    ├── vite.config.js
    ├── public/
    └── src/
        ├── App.jsx
        ├── api.js
        ├── main.jsx
        └── styles.css
```

## Backend Setup

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI:

```bash
uvicorn backend.main:app --reload
```

Backend docs:

```text
http://localhost:8000/docs
```

## Frontend Setup

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

If the backend is not on `http://localhost:8000`, create `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

## PDF Support

PDF upload uses `pdf2image`, which requires Poppler.

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
3. Add this folder to PATH:

```text
C:\poppler\Library\bin
```

Or set:

```powershell
$env:POPPLER_PATH="C:\poppler\Library\bin"
```

## API Endpoints

### `POST /upload`

Uploads a newspaper image/PDF, detects article boxes, and returns the resized
image URL plus bounding boxes.

Response example:

```json
{
  "session_id": "abc123",
  "image_url": "/image/abc123",
  "width": 1200,
  "height": 1600,
  "articles": [
    { "id": 1, "x": 120, "y": 80, "w": 400, "h": 500, "confidence": 0.72 }
  ]
}
```

### `GET /articles?session_id=abc123`

Returns article boxes for an existing session.

### `POST /ocr`

Runs OCR only on the selected article crop.

Form fields:

- `session_id`
- `article_id`
- `language`

### `POST /tts`

Generates MP3 audio for extracted text.

Form fields:

- `session_id`
- `article_id`
- `language`
- `text`

### `DELETE /session/{session_id}`

Deletes temporary session files.

## User Flow

```text
Upload newspaper image/PDF
-> Backend detects article regions
-> Frontend displays newspaper on canvas
-> Mouse hover finds article under cursor
-> Canvas draws glowing highlight
-> User clicks highlighted article
-> Frontend sends selected article ID to backend
-> Backend OCRs only selected crop
-> Text appears in panel
-> User optionally generates audio
-> Audio can be played/downloaded
```

## Notes

- PaddleOCR may download models on first use.
- gTTS requires internet access.
- The app avoids full-newspaper OCR for performance.
- Article detection is OpenCV-based. Very complex layouts or heavy ads may need
  tuning in `backend/utils/detector.py`.
