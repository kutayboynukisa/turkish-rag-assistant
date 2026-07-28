from turkish_rag_assistant.cleaner import clean_text


def test_clean_text_collapses_spaces():
    text = "hello      world"
    assert clean_text(text) == "hello world"


def test_clean_text_strips_edges():
    text = "   hello world   "
    assert clean_text(text) == "hello world"


def test_clean_text_collapses_blank_lines():
    text = "first line\n\n\n\nsecond line"
    assert clean_text(text) == "first line\n\nsecond line"


def test_clean_text_empty():
    assert clean_text("") == ""


def test_clean_text_handles_unicode_whitespace():
    text = "hello\u00a0\u00a0world"
    assert clean_text(text) == "hello world"
