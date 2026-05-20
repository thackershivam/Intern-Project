"""Optional text-to-speech generation."""

from __future__ import annotations

from pathlib import Path


TTS_LANGUAGES = {
    "Gujarati": "gu",
    "Hindi": "hi",
    "English": "en",
}


class TTSError(RuntimeError):
    """Raised when speech generation fails."""


def generate_speech(text: str, language: str, output_path: str | Path) -> Path:
    """Generate MP3 speech for extracted article text."""
    if not text.strip():
        raise TTSError("Cannot generate audio for empty text.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        from gtts import gTTS

        gTTS(text=text, lang=TTS_LANGUAGES.get(language, "gu")).save(str(destination))
    except Exception as exc:
        raise TTSError("gTTS failed to generate audio.") from exc

    return destination
