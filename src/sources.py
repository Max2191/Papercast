from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

_ARXIV_ID = re.compile(
    r"^(?:arxiv:)?(?P<id>(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7})(?:v\d+)?)$",
    re.IGNORECASE,
)
_ARXIV_URL = re.compile(
    r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf)/(?P<id>[^?#]+?)(?:\.pdf)?(?:[?#].*)?$",
    re.IGNORECASE,
)


def normalize_arxiv_reference(value: str) -> str | None:
    """Return a canonical arXiv PDF URL, or None when value is not arXiv."""

    cleaned = value.strip()
    direct_match = _ARXIV_ID.match(cleaned)
    if direct_match:
        return f"https://arxiv.org/pdf/{direct_match.group('id')}"

    url_match = _ARXIV_URL.match(cleaned)
    if url_match:
        paper_id = url_match.group("id").removesuffix(".pdf")
        return f"https://arxiv.org/pdf/{paper_id}"

    return None


def resolve_source(uploaded_file: str | Path | None, source_text: str | None) -> str:
    """Resolve an uploaded PDF or an arXiv/HTTP source into a Docling input."""

    if uploaded_file:
        path = Path(uploaded_file)
        if not path.exists():
            raise FileNotFoundError(f"Uploaded file does not exist: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError("The uploaded file must be a PDF.")
        return str(path)

    if not source_text or not source_text.strip():
        raise ValueError("Upload a PDF or enter an arXiv ID/URL.")

    source_text = source_text.strip()
    arxiv_url = normalize_arxiv_reference(source_text)
    if arxiv_url:
        return arxiv_url

    parsed = urlparse(source_text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return source_text

    raise ValueError("Enter a valid arXiv ID, arXiv URL, HTTP URL, or upload a PDF.")
