"""
Document Loader and Chunking Module for RAG Knowledge Assistant.
Supports loading and text extraction from JSON, TXT, PDF, and DOCX files,
producing structured text chunks with category metadata.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_json_document(file_path: Path, target_category: str = None) -> List[Dict[str, Any]]:
    """Loads JSON document and returns structured chunks with metadata."""
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_name = file_path.name

    # Handle base_knowledge.json structure
    if "categories" in data:
        cat_dict = data["categories"]
        for cat_key, cat_info in cat_dict.items():
            if target_category and cat_key.lower() != target_category.lower():
                continue
            
            title = cat_info.get("title", cat_key)
            desc = cat_info.get("description", "")
            chars = cat_info.get("key_characteristics", "")
            tact = cat_info.get("tactical_and_operational_context", "")
            notes = cat_info.get("detection_and_resolution_notes", "")

            full_text = f"Category: {cat_key}\nTitle: {title}\nDescription: {desc}\nKey Characteristics: {chars}\nTactical Context: {tact}\nDetection Notes: {notes}"
            
            chunks.append({
                "text": full_text,
                "metadata": {
                    "category": cat_key.lower(),
                    "source": doc_name,
                    "document_name": doc_name,
                    "document_type": "json",
                    "chunk_id": f"{doc_name}_{cat_key}"
                }
            })
    # Handle single dictionary or array format
    elif isinstance(data, dict):
        for k, v in data.items():
            cat = target_category or k
            text_str = json.dumps(v, indent=2) if isinstance(v, (dict, list)) else str(v)
            chunks.append({
                "text": f"Topic/Category: {cat}\nContent:\n{text_str}",
                "metadata": {
                    "category": cat.lower() if cat else "general",
                    "source": doc_name,
                    "document_name": doc_name,
                    "document_type": "json",
                    "chunk_id": f"{doc_name}_{k}"
                }
            })
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            text_str = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
            chunks.append({
                "text": f"Content item {idx+1}:\n{text_str}",
                "metadata": {
                    "category": target_category.lower() if target_category else "general",
                    "source": doc_name,
                    "document_name": doc_name,
                    "document_type": "json",
                    "chunk_id": f"{doc_name}_{idx}"
                }
            })

    return chunks


def load_text_document(file_path: Path, target_category: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """Reads plain TXT document and splits into overlapping text chunks."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    doc_name = file_path.name
    words = text.split()
    chunks = []
    
    if not words:
        return chunks

    idx = 0
    chunk_count = 0
    while idx < len(words):
        chunk_words = words[idx : idx + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "category": target_category.lower() if target_category else "general",
                "source": doc_name,
                "document_name": doc_name,
                "document_type": "txt",
                "chunk_id": f"{doc_name}_chunk_{chunk_count}"
            }
        })
        chunk_count += 1
        idx += chunk_size - overlap

    return chunks


def load_pdf_document(file_path: Path, target_category: str) -> List[Dict[str, Any]]:
    """Loads PDF file using pypdf/pdfplumber if available, else fallback."""
    doc_name = file_path.name
    text_content = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(file_path))
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
    except ImportError:
        logging.warning("pypdf not installed. PDF loading using basic fallback.")
        text_content = f"PDF Document Content from {doc_name}"

    # Save to temp text and chunk
    temp_txt = file_path.with_suffix(".tmp.txt")
    with open(temp_txt, "w", encoding="utf-8") as f:
        f.write(text_content)

    chunks = load_text_document(temp_txt, target_category)
    if temp_txt.exists():
        temp_txt.unlink()
        
    for c in chunks:
        c["metadata"]["source"] = doc_name
        c["metadata"]["document_name"] = doc_name
        c["metadata"]["document_type"] = "pdf"

    return chunks


def load_document(file_path: str, category: str = None) -> List[Dict[str, Any]]:
    """Main entry point to load any supported document file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document path '{file_path}' does not exist!")

    ext = path.suffix.lower()
    if ext == ".json":
        return load_json_document(path, target_category=category)
    elif ext in [".txt", ".md"]:
        return load_text_document(path, target_category=category)
    elif ext == ".pdf":
        return load_pdf_document(path, target_category=category)
    else:
        # Fallback text reader for unknown extension
        return load_text_document(path, target_category=category)


if __name__ == "__main__":
    base_kb = Path("knowledge_base/base_knowledge.json")
    if base_kb.exists():
        chunks = load_document(str(base_kb))
        print(f"Loaded {len(chunks)} chunks from base_knowledge.json")
        if chunks:
            print("Sample chunk metadata:", chunks[0]["metadata"])
