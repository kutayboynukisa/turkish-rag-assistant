"""Embedding utilities for turning text into dense vectors."""

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-base"


class Embedder:
    """Wraps a sentence-transformer model for query and passage embedding.

    The e5 model family expects role prefixes: search queries are prefixed
    with "query: " and indexed documents with "passage: ". This class adds
    those prefixes so callers don't have to.

    The model is loaded once when the instance is created, since loading is
    expensive relative to encoding.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Load the embedding model.

        Args:
            model_name: Hugging Face model identifier to load.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query.

        Args:
            text: The query text.

        Returns:
            The query embedding as a list of floats.
        """
        return self._encode_one(f"query: {text}")

    def embed_passage(self, text: str) -> list[float]:
        """Embed a single document passage.

        Args:
            text: The passage text.

        Returns:
            The passage embedding as a list of floats.
        """
        return self._encode_one(f"passage: {text}")

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document passages in one batch.

        Args:
            texts: The passage texts.

        Returns:
            One embedding per input text, in the same order.
        """

        prefixed = [f"passage: {text}" for text in texts]
        vectors = self.model.encode(prefixed, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def _encode_one(self, prefixed_text: str) -> list[float]:
        """Encode a single already-prefixed string."""
        vector = self.model.encode(prefixed_text, normalize_embeddings=True)
        return vector.tolist()
