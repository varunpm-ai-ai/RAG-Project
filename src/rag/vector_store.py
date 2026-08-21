"""
Local Vector Store Module using FAISS / NumPy with Category Metadata Filtering.
Persists vector index and document metadata locally in rag_storage/.
"""

import os
import sys
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.rag.embeddings import EmbeddingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VectorStore:
    def __init__(self, storage_dir: str = "rag_storage", embedder: EmbeddingGenerator = None):
        self.storage_dir = Path(storage_dir).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or EmbeddingGenerator()

        self.index_file = self.storage_dir / "faiss_index.bin"
        self.meta_file = self.storage_dir / "metadata.json"
        self.pickle_file = self.storage_dir / "vectors.pkl"

        self.documents = []  # list of chunk dicts: {"text": str, "metadata": dict}
        self.vectors = None  # numpy array of shape (N, dim)
        self.faiss_index = None
        self.use_faiss = False

        try:
            import faiss
            self.faiss = faiss
            self.use_faiss = True
        except ImportError:
            logging.warning("FAISS library not installed. Using numpy cosine similarity store.")

        self.load_storage()

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Adds text chunks with metadata to vector store."""
        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        new_vecs = self.embedder.embed_texts(texts)

        if self.vectors is None or len(self.vectors) == 0:
            self.vectors = new_vecs
            self.documents = chunks
        else:
            self.vectors = np.vstack([self.vectors, new_vecs])
            self.documents.extend(chunks)

        if self.use_faiss:
            dim = new_vecs.shape[1]
            if self.faiss_index is None:
                self.faiss_index = self.faiss.IndexFlatIP(dim)
            self.faiss_index.add(new_vecs)

        self.save_storage()
        logging.info(f"Added {len(chunks)} chunks to vector store. Total index count: {len(self.documents)}")
        return len(chunks)

    def search(self, query: str, category: str = None, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Searches vector store for top_k relevant chunks.
        Optionally filters results by category metadata.
        """
        if not self.documents or self.vectors is None or len(self.vectors) == 0:
            return []

        query_vec = self.embedder.embed_query(query).reshape(1, -1)

        # Filter candidates by category if specified
        candidate_indices = []
        for idx, doc in enumerate(self.documents):
            doc_cat = doc.get("metadata", {}).get("category", "").lower()
            if category is None or category.strip() == "" or doc_cat == category.lower() or doc_cat == "general":
                candidate_indices.append(idx)

        if not candidate_indices:
            # Fallback to all indices if category filter yields 0 candidates
            candidate_indices = list(range(len(self.documents)))

        cand_vectors = self.vectors[candidate_indices]
        similarities = np.dot(cand_vectors, query_vec.T).squeeze(-1)

        # Sort candidate indices by highest similarity score
        sorted_rel_indices = np.argsort(-similarities)
        
        results = []
        for rel_i in sorted_rel_indices[:top_k]:
            actual_idx = candidate_indices[rel_i]
            score = float(similarities[rel_i])
            doc = self.documents[actual_idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": round(score, 4)
            })

        return results

    def save_storage(self):
        """Persists vector store to disk."""
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, indent=2)

        with open(self.pickle_file, "wb") as f:
            pickle.dump(self.vectors, f)

        if self.use_faiss and self.faiss_index is not None:
            self.faiss.write_index(self.faiss_index, str(self.index_file))

    def load_storage(self):
        """Loads vector store from disk if present."""
        if self.meta_file.exists() and self.pickle_file.exists():
            try:
                with open(self.meta_file, "r", encoding="utf-8") as f:
                    self.documents = json.load(f)

                with open(self.pickle_file, "rb") as f:
                    self.vectors = pickle.load(f)

                if self.use_faiss and self.index_file.exists():
                    self.faiss_index = self.faiss.read_index(str(self.index_file))

                logging.info(f"Loaded existing vector storage with {len(self.documents)} indexed chunks.")
            except Exception as e:
                logging.warning(f"Could not load existing vector storage: {e}")

    def clear(self):
        """Clears all indexed documents except base storage."""
        self.documents = []
        self.vectors = None
        self.faiss_index = None
        if self.meta_file.exists():
            self.meta_file.unlink()
        if self.pickle_file.exists():
            self.pickle_file.unlink()
        if self.index_file.exists():
            self.index_file.unlink()
        logging.info("Vector storage cleared.")


if __name__ == "__main__":
    vs = VectorStore()
    sample_chunks = [
        {"text": "Planes fly in the air and land on runways.", "metadata": {"category": "plane", "source": "test.txt"}},
        {"text": "Ships sail across oceans and dock at harbors.", "metadata": {"category": "ship", "source": "test.txt"}}
    ]
    vs.add_chunks(sample_chunks)
    res = vs.search("runway aircraft", category="plane")
    print("Search result:", res)
