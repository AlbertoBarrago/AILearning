"""Vector Store for handling embeddings and storage operations."""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def add_texts(self, texts: List[str]) -> None:
        """Add texts to the store and generate their embeddings.

        Args:
            texts: List of texts to add
        """
        self.texts.extend(texts)
        embeddings = self.model.encode(texts)
        self.embeddings.extend(embeddings)

    def similarity_search(self, query: str, k: int = 2) -> List[str]:
        """Search for similar texts using cosine similarity.

        Args:
            query: Query text to compare against
            k: Number of results to return

        Returns:
            List of similar texts
        """
        query_embedding = self.model.encode([query])[0]  # Should be a 1D array

        # Ensure query_embedding is the right shape
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding.flatten()

        similarities = []

        for i, embedding in enumerate(self.embeddings):
            # Ensure embedding is the right shape
            if len(embedding.shape) > 1:
                embedding = embedding.flatten()

            # Make sure both have the same dimensions
            if query_embedding.shape != embedding.shape:
                # Either resize or log an error
                print(f"Shape mismatch: query={query_embedding.shape}, stored={embedding.shape}")
                continue

            similarity = np.dot(query_embedding, embedding) / \
                         (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
            similarities.append((similarity, i))

        similarities.sort(reverse=True)
        return [self.texts[idx] for _, idx in similarities[:k]]

    def get_embedding(self, text: str) -> np.ndarray:
        """Get the embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Vector embedding of the text
        """
        return self.model.encode([text])[0]
