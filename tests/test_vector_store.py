import pytest

from turkish_rag_assistant.embedder import Embedder
from turkish_rag_assistant.vector_store import VectorStore


@pytest.fixture(scope="module")
def embedder():
    return Embedder()


@pytest.fixture
def store(embedder):
    return VectorStore(embedder)


def test_new_store_is_empty(store):
    assert store.count() == 0


def test_add_increases_count(store):
    store.add(["birinci parça", "ikinci parça"])
    assert store.count() == 2


def test_search_returns_most_relevant_chunk(store):
    store.add(
        [
            "Ankara, Türkiye Cumhuriyeti'nin başkentidir.",
            "Zeytinyağlı enginar için önce limon suyu hazırlanır.",
            "Python'da listeler sort metoduyla sıralanır.",
        ]
    )
    results = store.search("Türkiye'nin başkenti neresi?", top_k=1)

    assert len(results) == 1
    assert "Ankara" in results[0]


def test_search_respects_top_k(store):
    store.add(["birinci", "ikinci", "üçüncü", "dördüncü"])
    results = store.search("bir şey", top_k=2)
    assert len(results) == 2


def test_search_on_empty_store_returns_nothing(store):
    assert store.search("herhangi bir soru", top_k=3) == []
