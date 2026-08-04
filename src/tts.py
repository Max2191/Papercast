from __future__ import annotations

import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import SETTINGS


@lru_cache(maxsize=1)
def _kokoro_pipeline():
    if not SETTINGS.enable_kokoro:
        raise RuntimeError("Kokoro is disabled. Set ENABLE_KOKORO=1 to enable it.")
    from kokoro import KPipeline

    return KPipeline(lang_code="a")


def generate_audio(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> Path:
    """Generate a WAV narration using the open-weight Kokoro TTS model."""

    import soundfile as sf

    normalized = text.strip()
    if not normalized:
        raise ValueError("Narration text is empty.")
    normalized = normalized[: SETTINGS.max_audio_chars]

    pipeline = _kokoro_pipeline()
    audio_segments: list[np.ndarray] = []
    for _, _, audio in pipeline(normalized, voice=voice, speed=speed):
        audio_segments.append(np.asarray(audio, dtype=np.float32))

    if not audio_segments:
        raise RuntimeError("Kokoro returned no audio segments.")

    combined = np.concatenate(audio_segments)
    output_path = SETTINGS.output_dir / f"papercast-{uuid.uuid4().hex[:10]}.wav"
    sf.write(output_path, combined, 24000)
    return output_path
