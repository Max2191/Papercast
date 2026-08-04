from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .models import PaperDocument
from .sections import split_markdown_sections

ParserMode = Literal["standard", "granite"]


@lru_cache(maxsize=1)
def _standard_converter():
    from docling.document_converter import DocumentConverter

    return DocumentConverter()


@lru_cache(maxsize=1)
def _granite_converter():
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline

    pipeline_options = VlmPipelineOptions(
        vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def parse_document(
    source: str,
    parser_mode: ParserMode = "granite",
    page_limit: int = 3,
) -> PaperDocument:
    """Convert a PDF/URL into structured Markdown with Docling.

    Granite mode is the multimodal transformer path. Standard mode is a faster
    fallback and a useful baseline for later extraction-quality comparisons.
    """

    if parser_mode not in {"standard", "granite"}:
        raise ValueError(f"Unknown parser mode: {parser_mode}")
    if page_limit < 1:
        raise ValueError("page_limit must be at least 1")

    converter = _granite_converter() if parser_mode == "granite" else _standard_converter()

    started = time.perf_counter()
    result = converter.convert(
        source=source,
        page_range=(1, page_limit),
        raises_on_error=True,
    )
    elapsed = time.perf_counter() - started
    markdown = result.document.export_to_markdown()
    sections = split_markdown_sections(markdown)

    filename = Path(source).name if not source.startswith("http") else source
    metadata = {
        "display_name": filename,
        "pages_requested": page_limit,
        "conversion_seconds": round(elapsed, 2),
        "section_count": len(sections),
        "word_count": len(markdown.split()),
        "formula_markers": markdown.count("$$") + markdown.count("\\[") + markdown.count("\\("),
    }
    return PaperDocument(
        source=source,
        parser_mode=parser_mode,
        markdown=markdown,
        sections=sections,
        metadata=metadata,
    )
