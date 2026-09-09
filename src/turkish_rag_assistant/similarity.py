"""Vectors similarity metrics for comparing embeddings."""

import math


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute the cosine similarity between two vectors.

    Cosine similarity measures the angle between two vectors, ignoring their
    magnitudes. It ranges from -1 (opposite) through 0 (unrelated) to 1
    (identical direction).

    Args:
        a: The first vector.
        b: The second vector.

    Returns:
        The cosine similarity, between -1.0 and 1.0.

    Raises:
        ValueError: If the vectors have diffrent lenghts or are empty.
    """
    if len(a) != len(b):
        raise ValueError("vectors must have the same lenght")

    if not a or not b:
        raise ValueError("vectors cannot be empty")

    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("vectors cannot be zero vectors")

    return dot_product / (magnitude_a * magnitude_b)
