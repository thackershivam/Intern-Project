"""Text-to-speech generation with gTTS."""

from __future__ import annotations

from pathlib import Path


TTS_LANGUAGE_CODES = {
    "Gujarati": "gu",
    "Hindi": "hi",
    "English": "en",
}


class TTSError(RuntimeError):
    """Raised when audio generation fails."""


def generate_audio(
    text: str,
    language_label: str,
    output_path: str | Path = "audio/output.mp3",
    *,
    speed: str = "normal",
) -> Path:
    """Generate speech for OCR text in the selected language."""
    if not text.strip():
        raise TTSError("Cannot generate audio from empty text.")
    if speed not in {"normal", "slow"}:
        raise TTSError("Unsupported speech speed.")

    language_code = TTS_LANGUAGE_CODES.get(language_label, "gu")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        from gtts import gTTS

        gTTS(text=text, lang=language_code, slow=speed == "slow").save(str(destination))
    except Exception as exc:
        raise TTSError("Failed to generate audio with gTTS.") from exc

    return destination
