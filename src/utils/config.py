import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env or dotenv
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")
load_dotenv(ROOT_DIR / "dotenv")

# Available Free-Tier Multimodal Models (Active Google API Endpoint Models)
FREE_TIER_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
]

DEFAULT_MODEL = "gemini-3.6-flash"

SYSTEM_PERSONAS = {
    "General Assistant": "You are a helpful, friendly, and intelligent AI assistant capable of multimodal conversation.",
    "Visual Specialist": "You are an expert visual analyst. Analyze uploaded images in depth, describing objects, text (OCR), colors, layout, composition, spatial features, and context with exceptional precision.",
    "Technical & Code Expert": "You are a senior software engineer and AI researcher. Analyze technical diagrams, screenshots, architecture schemas, and answer questions with clean code examples and technical accuracy.",
    "Detailed Analyst": "Provide comprehensive, structured, step-by-step responses, breaking down visual and textual information into clear analytical findings."
}

def get_api_key() -> str:
    """Retrieve Gemini API Key from environment or session state."""
    api_key = (
        os.getenv("GEMINI_API_KEY") 
        or os.getenv("GEMINIAPI") 
        or os.getenv("GOOGLE_API_KEY")
    )
    if not api_key and "user_api_key" in st.session_state and st.session_state["user_api_key"]:
        api_key = st.session_state["user_api_key"].strip()
    return api_key or ""
