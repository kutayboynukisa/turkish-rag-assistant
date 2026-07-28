"""Text measurement utilities for computing basic document statistics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TextStats:
    word_count: int
    char_count: int
    line_count: int


def compute_text_stats(text: str) -> TextStats:
    """Compute the word, character, and line counts of a text.

    Word count is the number of whitespace-separated tokens, so hyphenated
    or punctuation-joined words count as one. Character count includes all
    characters, including whitespace.

    Args:
        text: The input text to measure.

    Returns:
        A TextStats with word, character, and line counts."""
    return TextStats(
        word_count=len(text.split()),
        char_count=len(text),
        line_count=len(text.splitlines()),
    )
