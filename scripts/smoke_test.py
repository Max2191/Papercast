"""Small local smoke test that does not download neural models."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.narration import prepare_narration
from src.sections import split_markdown_sections
from src.sources import normalize_arxiv_reference
from src.summarization import extractive_summary

sample = """
# Abstract
We introduce a document-understanding system for scientific papers. The system preserves formulas and structure.

# Method
A multimodal transformer converts document-page images into structured tokens. A sequence-to-sequence transformer then summarizes selected sections.

# Conclusion
The combined pipeline produces text suitable for a neural text-to-speech model.
"""

sections = split_markdown_sections(sample)
summary = extractive_summary(sections, "overview")
narration = prepare_narration(summary, "overview")

assert normalize_arxiv_reference("2501.17887")
assert sections
assert summary
assert narration
print("Smoke test passed.")
