"""
LLM Integration Module for Category-Specific RAG Knowledge Assistant.
Provides clean abstraction for Google Gemini API or local RAG synthesis fallback.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class LLMClient:
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model_name = model_name
        self.client = None
        self.is_api_available = False

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                self.is_api_available = True
                logging.info("Google GenAI LLM Client initialized successfully.")
            except Exception as e:
                logging.warning(f"Could not initialize Google GenAI Client: {e}")

    def generate_rag_response(
        self,
        question: str,
        detected_category: str,
        retrieved_chunks: list
    ) -> str:
        """
        Synthesizes an answer using the retrieved context chunks and detected object category.
        Instructs LLM to rely strictly on retrieved context.
        """
        if not retrieved_chunks:
            return (
                f"The knowledge base does not contain any indexed information for the detected object category '{detected_category}'. "
                "Please upload a reference document for this category to expand the knowledge base."
            )

        context_str = ""
        sources = set()
        for idx, chunk in enumerate(retrieved_chunks, 1):
            source_name = chunk.get("metadata", {}).get("source", "Unknown Source")
            sources.add(source_name)
            context_str += f"\n--- Source {idx} [{source_name}] ---\n{chunk['text']}\n"

        system_prompt = (
            "You are an expert AI Assistant for aerial imagery intelligence and object detection.\n"
            f"Detected Category: {detected_category.upper()}\n"
            "Use ONLY the supplied retrieved context documents below as your primary knowledge source.\n"
            "Do NOT invent facts or hallucinate details not supported by the context.\n"
            "If the retrieved documents do not contain enough information to fully answer the question, "
            "explicitly state that the available knowledge base contains partial information."
        )

        user_prompt = (
            f"{system_prompt}\n\n"
            f"RETRIEVED CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "ANSWER:"
        )

        if self.is_api_available and self.client is not None:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt
                )
                return response.text
            except Exception as e:
                logging.warning(f"GenAI API call failed: {e}. Falling back to structured context synthesis.")

        # Fallback structured RAG answer generation (Deterministic for local/offline MVP demo)
        fallback_ans = f"### Summary based on Knowledge Base for '{detected_category.upper()}':\n\n"
        for chunk in retrieved_chunks:
            source = chunk.get('metadata', {}).get('source', 'Base Knowledge')
            text_snippet = chunk['text']
            fallback_ans += f"**Source ({source}):**\n{text_snippet}\n\n"
        
        return fallback_ans


if __name__ == "__main__":
    llm = LLMClient()
    dummy_chunks = [{
        "text": "Fixed-wing aircraft feature swept wings and operate on airport runways.",
        "metadata": {"source": "base_knowledge.json", "category": "plane"}
    }]
    ans = llm.generate_rag_response("What are planes?", "plane", dummy_chunks)
    print("Generated Answer:\n", ans)
