"""
High-Level RAG Pipeline Orchestrator.
Integrates Document Loader, Embedding Generator, Local Vector Store, and LLM Client.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.documents.loader import load_document
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.llm import LLMClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class RagPipeline:
    def __init__(self, storage_dir: str = "rag_storage", api_key: str = None):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore(storage_dir=storage_dir, embedder=self.embedder)
        self.llm_client = LLMClient(api_key=api_key)
        self.uploaded_documents = set()

        # Check if base knowledge needs initialization
        self.auto_init_base_knowledge()

    def auto_init_base_knowledge(self, base_kb_path: str = "knowledge_base/base_knowledge.json"):
        """Loads default base knowledge JSON if vector store is currently empty."""
        kb_file = Path(base_kb_path).resolve()
        if kb_file.exists() and len(self.vector_store.documents) == 0:
            logging.info(f"Auto-initializing base knowledge from: '{kb_file}'")
            chunks = load_document(str(kb_file))
            self.vector_store.add_chunks(chunks)

    def index_document(self, file_path: str, category: str) -> int:
        """Indexes a new user-uploaded document for a specific category."""
        path = Path(file_path)
        chunks = load_document(str(path), category=category)
        added_count = self.vector_store.add_chunks(chunks)
        if added_count > 0:
            self.uploaded_documents.add(path.name)
        return added_count

    def answer_question(self, question: str, category: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Retrieves category-filtered context chunks and generates answer via LLM.
        """
        retrieved_chunks = self.vector_store.search(
            query=question,
            category=category,
            top_k=top_k
        )

        answer = self.llm_client.generate_rag_response(
            question=question,
            detected_category=category,
            retrieved_chunks=retrieved_chunks
        )

        sources_used = []
        seen = set()
        for chunk in retrieved_chunks:
            src = chunk.get("metadata", {}).get("source", "Base Knowledge")
            if src not in seen:
                sources_used.append(src)
                seen.add(src)

        return {
            "question": question,
            "category": category,
            "answer": answer,
            "sources": sources_used,
            "retrieved_chunks": retrieved_chunks
        }

    def get_stats(self) -> Dict[str, Any]:
        """Returns knowledge base statistics."""
        docs = self.vector_store.documents
        sources = set([d.get("metadata", {}).get("source", "unknown") for d in docs])
        categories = set([d.get("metadata", {}).get("category", "unknown") for d in docs])
        
        return {
            "total_chunks": len(docs),
            "sources_count": len(sources),
            "sources_list": list(sources),
            "categories_count": len(categories)
        }

    def reset_storage(self):
        """Resets vector store and re-initializes base knowledge."""
        self.vector_store.clear()
        self.uploaded_documents.clear()
        self.auto_init_base_knowledge()


if __name__ == "__main__":
    pipeline = RagPipeline()
    stats = pipeline.get_stats()
    print("RAG Pipeline Stats:", stats)
    res = pipeline.answer_question("Tell me about aircraft", category="plane")
    print("Sample QA Answer:\n", res["answer"][:200])
