"""Text cleaning utilities for normalizing whitespace in raw documents."""

import re

_MULTISPACE = re.compile(r"[\s]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Clean raw text by normalizing whitespace and blank lines.

    Collapses runs of whitespace (including Unicode whitespace such as
    non-breaking spaces) into single spaces, strips leading and trailing
    whitespace from each line, and collapses three or more consecutive
    newlines into a single blank line.

    Args:
        text: The raw input text to clean.

    Returns:
        The cleaned text."""
    lines = text.splitlines()

    cleaned_lines = []
    for line in lines:
        collapsed = _MULTISPACE.sub(" ", line)
        cleaned_lines.append(collapsed.strip())

    joined = "\n".join(cleaned_lines)
    joined = _BLANK_LINES.sub("\n\n", joined)

    return joined.strip()
