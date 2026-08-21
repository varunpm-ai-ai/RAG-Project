"""
Embeddings Generator Module for RAG Knowledge Assistant.
Uses SentenceTransformers ('all-MiniLM-L6-v2') for semantic vector embeddings
with a robust fallback to TF-IDF vectorizer if neural models are unavailable.
"""

import logging
import numpy as np
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = 384
        self.is_fallback = False

        try:
            from sentence_transformers import SentenceTransformer
            logging.info(f"Loading embedding model: '{model_name}'...")
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logging.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logging.warning(f"Could not load SentenceTransformer ('{e}'). Using TF-IDF vectorizer fallback.")
            self.is_fallback = True
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=384)
            self.is_fitted = False

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generates normalized vector embeddings for a list of text strings."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        if not self.is_fallback and self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            # Normalize vectors L2 norm
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (embeddings / norms).astype(np.float32)
        else:
            # Fallback TF-IDF representation
            if not self.is_fitted:
                matrix = self.vectorizer.fit_transform(texts).toarray()
                self.is_fitted = True
            else:
                matrix = self.vectorizer.transform(texts).toarray()
            
            # Pad or truncate to 384 dimensions
            if matrix.shape[1] < 384:
                pad_width = 384 - matrix.shape[1]
                matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
            elif matrix.shape[1] > 384:
                matrix = matrix[:, :384]

            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return (matrix / norms).astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Generates embedding vector for a single search query."""
        vecs = self.embed_texts([query])
        return vecs[0]


if __name__ == "__main__":
    embedder = EmbeddingGenerator()
    vecs = embedder.embed_texts(["Aircraft runway analysis", "Ship maritime tracking"])
    print(f"Generated embeddings shape: {vecs.shape}")
