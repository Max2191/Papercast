from __future__ import annotations

import re

_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_LATEX_BLOCK = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_LATEX_INLINE = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")


def _verbalize_math(match: re.Match[str]) -> str:
    expression = " ".join(match.group(1).split())
    expression = expression.replace("\\", " ")
    expression = expression.replace("_", " sub ").replace("^", " to the power of ")
    expression = expression.replace("{", " ").replace("}", " ")
    return f" The paper gives the mathematical expression: {expression}. "


def prepare_narration(summary: str, technicality: str) -> str:
    """Convert a written summary into a speech-friendly script.

    This first version performs deterministic NLP cleanup. A later milestone can
    add a generative 'explain this equation conceptually' model.
    """

    cleaned = _MARKDOWN_LINK.sub(r"\1", summary)
    cleaned = _INLINE_CODE.sub(r"\1", cleaned)
    cleaned = _HEADING.sub("", cleaned)
    cleaned = _LATEX_BLOCK.sub(_verbalize_math, cleaned)
    cleaned = _LATEX_INLINE.sub(_verbalize_math, cleaned)
    cleaned = cleaned.replace("*", "").replace("#", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    intros = {
        "overview": "Here is a concise overview of the paper. ",
        "intermediate": "Here is an explanatory walkthrough of the paper. ",
        "technical": "Here is a technical briefing on the paper. ",
    }
    outro = " That concludes the PaperCast briefing."
    return intros.get(technicality, intros["intermediate"]) + cleaned + outro
