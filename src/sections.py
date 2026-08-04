from __future__ import annotations

import re

from .models import Section

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def split_markdown_sections(markdown: str) -> list[Section]:
    """Split Markdown into heading-aware sections while preserving source order."""

    lines = markdown.splitlines()
    sections: list[Section] = []
    current_title = "Document opening"
    current_level = 1
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append(
                Section(title=current_title, level=current_level, content=content)
            )
        current_lines = []

    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            flush()
            current_level = len(heading.group(1))
            current_title = heading.group(2).strip()
        else:
            current_lines.append(line)

    flush()
    return sections


def section_table(sections: list[Section]) -> list[list[object]]:
    return [[index + 1, item.level, item.title, item.word_count] for index, item in enumerate(sections)]
