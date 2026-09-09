import pytest

from turkish_rag_assistant.embedder import Embedder
from turkish_rag_assistant.similarity import cosine_similarity


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


def test_related_passage_scores_higher_than_unrelated(embedder):
    query = embedder.embed_query("Türkiye'nin başkenti neresi?")

    related = embedder.embed_passage(
        "Ankara, Türkiye Cumhuriyeti'nin idari merkezidir."
    )

    unraleted = embedder.embed_passage(
        "Zeytinyağlı enginer yemeği için önce limon suyu hazırlanır."
    )

    related_score = cosine_similarity(query, related)
    unraleted_score = cosine_similarity(query, unraleted)

    assert related_score > unraleted_score


def test_paraphrases_are_more_similar_than_different_topics(embedder):
    original = embedder.embed_passage("Kedi bahçede uyuyor.")
    paraphrase = embedder.embed_passage("Bahçedeki kedi uykuya dalmış.")
    different = embedder.embed_passage("Borsa endeksi bugün yüzde iki düştü.")

    assert cosine_similarity(original, paraphrase) > cosine_similarity(
        original, different
    )


def test_query_and_passage_of_same_topic_are_reasonably_similar(embedder):
    query = embedder.embed_query("Python'da liste nasıl sıralanır?")
    passage = embedder.embed_passage(
        "Python listelerini sıralamak için sort metodu kullanılır."
    )

    assert cosine_similarity(query, passage) > 0.7
