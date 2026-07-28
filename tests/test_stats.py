from turkish_rag_assistant.stats import compute_text_stats


def test_compute_text_stats_basic():
    text = "hello world\nhow are you"
    stats = compute_text_stats(text)

    assert stats.word_count == 5
    assert stats.char_count == 23
    assert stats.line_count == 2


def test_compute_text_stats_empty():
    stats = compute_text_stats("")
    assert stats.word_count == 0
    assert stats.char_count == 0
    assert stats.line_count == 0
