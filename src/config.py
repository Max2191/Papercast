from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "PaperCast"
    granite_model_id: str = os.getenv(
        "GRANITE_MODEL_ID", "ibm-granite/granite-docling-258M"
    )
    summarizer_model_id: str = os.getenv(
        "SUMMARIZER_MODEL_ID", "google/pegasus-x-base-arxiv"
    )

    summarizer_tokenizer_id: str = os.getenv(
    "SUMMARIZER_TOKENIZER_ID", "google/pegasus-x-base"
)
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "outputs"))
    max_audio_chars: int = int(os.getenv("MAX_AUDIO_CHARS", "12000"))
    enable_pegasus: bool = os.getenv("ENABLE_PEGASUS", "1") == "1"
    enable_kokoro: bool = os.getenv("ENABLE_KOKORO", "1") == "1"


SETTINGS = Settings()
SETTINGS.output_dir.mkdir(parents=True, exist_ok=True)
