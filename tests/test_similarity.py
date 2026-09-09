import pytest

from turkish_rag_assistant.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    vector = [0.6, 0.8]
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_minus_one():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_rejects_vectors_of_different_lenght():
    with pytest.raises(ValueError, match="same lenght"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


def test_rejects_empty_vectors():
    with pytest.raises(ValueError, match="cannot be empty"):
        cosine_similarity([], [])
