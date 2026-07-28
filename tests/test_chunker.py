import pytest

from turkish_rag_assistant.chunker import chunk_text


def test_chunk_text_splits_into_sizes():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=5, overlap=0)
    assert chunks == ["abcde", "fghij"]


def test_chunk_text_with_overlap():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == ["abcde", "defgh", "ghij"]


def test_chunk_text_shorter_than_size():
    text = "abc"
    chunks = chunk_text(text, chunk_size=5, overlap=0)
    assert chunks == ["abc"]


def test_chunk_text_rejects_zero_chunk_size():
    with pytest.raises(ValueError, match="positive integer"):
        chunk_text("abc", chunk_size=0, overlap=0)


def test_chunk_text_rejects_negative_overlap():
    with pytest.raises(ValueError, match="cannot be negative"):
        chunk_text("abc", chunk_size=5, overlap=-1)


def test_chunk_text_rejects_overlap_larger_than_chunk_size():
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_text("abc", chunk_size=5, overlap=5)


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=5, overlap=0) == []
