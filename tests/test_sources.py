from pathlib import Path

import pytest

from src.sources import normalize_arxiv_reference, resolve_source


def test_normalize_arxiv_id():
    assert normalize_arxiv_reference("2501.17887") == "https://arxiv.org/pdf/2501.17887"


def test_normalize_arxiv_url():
    assert normalize_arxiv_reference("https://arxiv.org/abs/2501.17887") == "https://arxiv.org/pdf/2501.17887"


def test_rejects_invalid_source():
    with pytest.raises(ValueError):
        resolve_source(None, "not a source")


def test_local_pdf(tmp_path: Path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-test")
    assert resolve_source(paper, None) == str(paper)
