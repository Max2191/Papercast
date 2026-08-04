from src.sections import split_markdown_sections


def test_split_sections():
    markdown = """Opening text.\n\n# Introduction\nIntro content.\n\n## Method\nMethod content."""
    sections = split_markdown_sections(markdown)
    assert [section.title for section in sections] == ["Document opening", "Introduction", "Method"]
    assert sections[-1].level == 2
