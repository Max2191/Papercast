from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Literal

from .config import SETTINGS
from .models import Section

Technicality = Literal["overview", "intermediate", "technical"]
SummarizerMode = Literal["extractive", "pegasus"]

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_STOPWORDS = {
    "the", "and", "that", "with", "from", "this", "were", "have", "has", "had",
    "for", "are", "was", "but", "not", "into", "their", "they", "our", "using",
    "used", "than", "then", "which", "also", "can", "may", "such", "these",
    "those", "between", "within", "where", "when", "while", "about", "paper",
}


def _preferred_text(sections: list[Section], technicality: Technicality) -> str:
    if not sections:
        return ""

    preferred_terms = {
        "overview": ("abstract", "introduction", "conclusion", "discussion"),
        "intermediate": ("abstract", "introduction", "method", "result", "conclusion", "discussion"),
        "technical": (),
    }[technicality]

    if not preferred_terms:
        return "\n\n".join(section.content for section in sections)

    selected = [
        section.content
        for section in sections
        if any(term in section.title.lower() for term in preferred_terms)
    ]
    return "\n\n".join(selected or [section.content for section in sections])


def extractive_summary(sections: list[Section], technicality: Technicality) -> str:
    """Transparent baseline summarizer used when the neural model is unavailable."""

    text = _preferred_text(sections, technicality)
    sentences = [item.strip() for item in _SENTENCE.split(text) if len(item.split()) >= 7]
    if not sentences:
        return text[:3000].strip()

    words = [word.lower() for word in _WORD.findall(text)]
    frequencies = Counter(word for word in words if word not in _STOPWORDS)
    if not frequencies:
        return " ".join(sentences[:8])

    max_frequency = max(frequencies.values())
    normalized = {word: count / max_frequency for word, count in frequencies.items()}

    scored: list[tuple[int, float, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_words = [word.lower() for word in _WORD.findall(sentence)]
        if not sentence_words:
            continue
        score = sum(normalized.get(word, 0.0) for word in sentence_words)
        score /= math.sqrt(len(sentence_words))
        scored.append((index, score, sentence))

    target = {"overview": 8, "intermediate": 14, "technical": 22}[technicality]
    chosen = sorted(sorted(scored, key=lambda item: item[1], reverse=True)[:target])
    return "\n\n".join(sentence for _, _, sentence in chosen)


@lru_cache(maxsize=1)
def _load_pegasus():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        SETTINGS.summarizer_tokenizer_id,
        use_fast=False,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(SETTINGS.summarizer_model_id)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _token_chunks(tokenizer, text: str, chunk_size: int) -> list[list[int]]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [token_ids[index:index + chunk_size] for index in range(0, len(token_ids), chunk_size)]


def pegasus_summary(sections: list[Section], technicality: Technicality) -> str:
    """Abstractive scientific summarization with a pretrained PEGASUS-X model."""

    if not SETTINGS.enable_pegasus:
        raise RuntimeError("PEGASUS is disabled. Set ENABLE_PEGASUS=1 to enable it.")

    import torch

    text = _preferred_text(sections, technicality)
    if not text.strip():
        raise ValueError("No paper text is available to summarize.")

    tokenizer, model, device = _load_pegasus()
    model_limit = getattr(tokenizer, "model_max_length", 4096)
    if not isinstance(model_limit, int) or model_limit > 32768:
        model_limit = 4096
    chunk_size = min(model_limit - 64, 4096)

    length_settings = {
        "overview": (96, 260),
        "intermediate": (160, 420),
        "technical": (240, 620),
    }
    min_new_tokens, max_new_tokens = length_settings[technicality]

    partial_summaries: list[str] = []
    for ids in _token_chunks(tokenizer, text, chunk_size):
        input_ids = torch.tensor([ids], device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True,
            )
        partial_summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    merged = "\n\n".join(partial_summaries)
    merged_ids = tokenizer.encode(merged, add_special_tokens=False)[:chunk_size]
    input_ids = torch.tensor([merged_ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=4,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def summarize(
    sections: list[Section],
    technicality: Technicality,
    mode: SummarizerMode,
) -> str:
    if mode == "extractive":
        return extractive_summary(sections, technicality)
    if mode == "pegasus":
        return pegasus_summary(sections, technicality)
    raise ValueError(f"Unknown summarizer mode: {mode}")
