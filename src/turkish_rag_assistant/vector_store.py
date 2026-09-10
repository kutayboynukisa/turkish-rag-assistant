"""In-memory vector store backed by Chroma for semantic retrieval."""

import uuid

import chromadb

from turkish_rag_assistant.embedder import Embedder


class VectorStore:
    """Stores text chunks and retrieves them by semantic similarity.

    Embedding are produced by the injected Embedder rather than by Chroma,
    so the Turkish-tuned model and its query/passage prefixes stay in effect.

    Args:
        embedder: The embedder used for both indexing and querying.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name=f"chunks_{uuid.uuid4().hex}"
        )

    def add(self, chunks: list[str]) -> None:
        """Embed and store text chunks.

        Args:
            chunks: The text chunks to index.
        """
        if not chunks:
            return

        vectors = self.embedder.embed_passages(chunks)
        ids = [uuid.uuid4().hex for _ in chunks]

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=vectors,
        )

    def search(self, query: str, top_k: int = 5) -> list[str]:
        """Find the chunks most similar to a query.

        Args:
            query: The search query.
            top_k: Maximum number of chunks to return.

        Returns:
            The matching chunk texts, most similar first.

        Raises:
            ValueError: If top_k is not positive.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if self.count() == 0:
            return []

        query_vector = self.embedder.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
        )

        return results["documents"][0]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()
