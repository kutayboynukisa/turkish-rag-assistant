import pytest

from turkish_rag_assistant.embedder import Embedder


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_embed_passage_returns_correct_dimension(embedder):
    vector = embedder.embed_passage("Ankara Türkiye'nin başkentidir.")
    assert len(vector) == 768


def test_embed_query_returns_correct_dimension(embedder):
    vector = embedder.embed_query("Türkiye'nin başkenti neresi?")
    assert len(vector) == 768


def test_embed_passages_return_one_vector_per_text(embedder):
    vectors = embedder.embed_passages(["birinci metin", "ikinci metin"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 768
