from src.narration import prepare_narration


def test_narration_removes_markdown_and_verbalizes_math():
    text = "# Result\nThe state obeys $H|psi> = E|psi>$ and [details](https://example.com)."
    narration = prepare_narration(text, "technical")
    assert "#" not in narration
    assert "https://" not in narration
    assert "mathematical expression" in narration
