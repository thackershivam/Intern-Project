"""Text-to-speech generation for summarized newspaper output."""

from __future__ import annotations

from pathlib import Path


LANGUAGE_CODES = {
    "gujarati": "gu",
    "hindi": "hi",
    "english": "en",
}

VOICE_SPEEDS = {"normal", "slow"}


class TTSError(RuntimeError):
    """Raised when audio generation fails."""


def generate_audio_summary(
    summary_text: str,
    *,
    output_path: str | Path = "audio/summary.mp3",
    language: str = "gujarati",
    speed: str = "normal",
) -> Path:
    """Convert summary text into an MP3 file using gTTS."""
    if not summary_text.strip():
        raise TTSError("Cannot generate audio from empty summary text.")

    normalized_language = language.lower().strip()
    normalized_speed = speed.lower().strip()
    if normalized_speed not in VOICE_SPEEDS:
        raise TTSError("Unsupported voice speed. Choose 'normal' or 'slow'.")

    lang_code = LANGUAGE_CODES.get(normalized_language, "gu")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        from gtts import gTTS

        tts = gTTS(
            text=summary_text,
            lang=lang_code,
            slow=normalized_speed == "slow",
        )
        tts.save(str(destination))
    except Exception as exc:
        raise TTSError("Failed to generate Gujarati audio summary.") from exc

    return destination
