from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Section:
    """A Markdown section extracted from a converted paper."""

    title: str
    level: int
    content: str

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PaperDocument:
    """Application-friendly representation of a processed paper."""

    source: str
    parser_mode: str
    markdown: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "parser_mode": self.parser_mode,
            "markdown": self.markdown,
            "sections": [section.to_dict() for section in self.sections],
            "metadata": self.metadata,
            "summary": "",
            "narration": "",
        }
