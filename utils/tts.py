"""Gujarati text-to-speech generation."""

from __future__ import annotations

from pathlib import Path


class TTSError(RuntimeError):
    """Raised when speech generation fails."""


def generate_gujarati_audio(
    text: str,
    output_path: str | Path,
    *,
    speed: str = "normal",
) -> Path:
    """Generate an MP3 file for selected Gujarati article text."""
    if not text.strip():
        raise TTSError("Cannot generate audio from empty article text.")
    if speed not in {"normal", "slow"}:
        raise TTSError("Unsupported speed. Choose normal or slow.")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        from gtts import gTTS

        speech = gTTS(text=text, lang="gu", slow=speed == "slow")
        speech.save(str(destination))
    except Exception as exc:
        raise TTSError("Failed to generate Gujarati audio.") from exc

    return destination
