"""Utilities for splitting cleaned text into overlapping chunks."""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size character chunks with optional overlap.

    The text is split on raw character offsets, so a chunk may cut through
    the middle of a word or sentence. chunk_size is an exact character count,
    not an approximate target.

    Args:
        text: The input text to split.
        chunk_size: Number of characters per chunk. Must be positive.
        overlap: Number of characters each chunk shares with the previous
            one. Must be non-negative and smaller than chunk_size.

    Returns:
        A list of text chunks. Empty input returns an empty list.

    Raises:
        ValueError: If chunk_size is not positive, overlap is negative,
            or overlap is greater than or equal to chunk_size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks
