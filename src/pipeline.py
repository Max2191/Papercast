from __future__ import annotations

from .models import PaperDocument, Section
from .narration import prepare_narration
from .parsing import parse_document
from .sources import resolve_source
from .summarization import summarize
from .tts import generate_audio


def process_paper(
    uploaded_file: str | None,
    source_text: str | None,
    parser_mode: str,
    page_limit: int,
) -> PaperDocument:
    source = resolve_source(uploaded_file, source_text)
    return parse_document(source, parser_mode=parser_mode, page_limit=page_limit)


def sections_from_state(state: dict) -> list[Section]:
    return [Section(**item) for item in state.get("sections", [])]


def summarize_state(state: dict, technicality: str, mode: str) -> tuple[dict, str]:
    sections = sections_from_state(state)
    if not sections:
        raise ValueError("Process a paper before generating a summary.")
    summary = summarize(sections, technicality=technicality, mode=mode)
    updated = dict(state)
    updated["summary"] = summary
    updated["technicality"] = technicality
    updated["summarizer_mode"] = mode
    return updated, summary


def narration_from_state(state: dict) -> tuple[dict, str]:
    summary = state.get("summary", "").strip()
    if not summary:
        raise ValueError("Generate a summary before preparing narration.")
    technicality = state.get("technicality", "intermediate")
    narration = prepare_narration(summary, technicality)
    updated = dict(state)
    updated["narration"] = narration
    return updated, narration


def audio_from_state(state: dict, voice: str, speed: float) -> str:
    narration = state.get("narration", "").strip()
    if not narration:
        raise ValueError("Prepare narration before generating audio.")
    return str(generate_audio(narration, voice=voice, speed=speed))
