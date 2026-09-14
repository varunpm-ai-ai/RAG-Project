import streamlit as st
from PIL import Image
import os

from src.utils.config import (
    get_api_key, 
    FREE_TIER_MODELS, 
    DEFAULT_MODEL, 
    SYSTEM_PERSONAS
)
from src.ai.gemini_service import GeminiService

# Page Setup
st.set_page_config(
    page_title="Multimodal Gemini AI Assistant",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Modern Generative AI App & Mobile Responsiveness
st.markdown("""
    <style>
    /* Main layout responsiveness */
    .main .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 960px;
    }
    
    /* Responsive Chat Bubbles */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.75rem;
        word-break: break-word;
    }
    
    /* Responsive Images */
    .stChatMessage img, .stImage img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 8px;
    }
    
    /* Mobile specific adjustments */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 1rem !important;
        }
        .stTitle {
            font-size: 1.6rem !important;
        }
        .stSubheader {
            font-size: 1.1rem !important;
        }
    }
    
    /* Active Image Card */
    .active-image-card {
        border-left: 4px solid #1A73E8;
        background-color: rgba(26, 115, 232, 0.08);
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "active_image" not in st.session_state:
    st.session_state["active_image"] = None

if "image_name" not in st.session_state:
    st.session_state["image_name"] = None

def reset_chat():
    """Clear session state for a fresh new chat."""
    st.session_state["messages"] = []
    st.session_state["active_image"] = None
    st.session_state["image_name"] = None

# Sidebar Setup
with st.sidebar:
    st.title(":material/auto_awesome: Gemini AI Assistant")
    st.caption("Round 2 • Multimodal Generative AI")

    # New Chat Button
    if st.button(":material/add: New Chat", type="primary", width="stretch"):
        reset_chat()
        st.rerun()

    st.divider()

    # API Key Configuration
    current_key = get_api_key()
    if current_key:
        st.success(":material/key: API Key Loaded", icon="✅")
        api_key = current_key
    else:
        st.warning(":material/key_off: API Key Required")
        api_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            key="user_api_key",
            help="Added in .env or enter your key from Google AI Studio."
        )

    # Model Selection (Free Tier Compliant)
    selected_model = st.selectbox(
        "AI Model (Free Tier)",
        options=FREE_TIER_MODELS,
        index=0,
        help="Recommended: gemini-3.6-flash or gemini-3.5-flash for fast multimodal reasoning."
    )

    # System Persona Selector
    selected_persona_name = st.selectbox(
        "AI Persona",
        options=list(SYSTEM_PERSONAS.keys()),
        index=0
    )
    system_instruction = SYSTEM_PERSONAS[selected_persona_name]

    st.divider()

    # Image Uploader Section
    st.markdown("### :material/image: Image Upload")
    uploaded_file = st.file_uploader(
        "Upload Image to Discuss",
        type=["png", "jpg", "jpeg", "webp"],
        key="sidebar_uploader",
        help="Upload an image to ask questions, analyze visual data, or extract text."
    )

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.session_state["active_image"] = img
            st.session_state["image_name"] = uploaded_file.name
        except Exception as e:
            st.error(f"Failed to load image: {e}")

    # Active Image Preview in Sidebar
    if st.session_state["active_image"] is not None:
        st.image(
            st.session_state["active_image"], 
            caption=f"Attached: {st.session_state['image_name'] or 'Image'}", 
            width="stretch"
        )
        if st.button(":material/delete: Remove Image", width="stretch"):
            st.session_state["active_image"] = None
            st.session_state["image_name"] = None
            st.rerun()

# Main UI Header
st.title("💬 Talk with Gemini AI")
st.caption("Upload images and have natural conversations powered by Google Gemini API.")

# Display Active Image Banner in Main Chat if set
if st.session_state["active_image"] is not None:
    with st.container(border=True):
        col_img, col_info = st.columns([1, 4])
        with col_img:
            st.image(st.session_state["active_image"], width=120)
        with col_info:
            st.markdown(f"**:material/image: Active Image Attached**: `{st.session_state['image_name'] or 'Uploaded Image'}`")
            st.caption("Ask any question below about this image. The AI will analyze it in context.")

# Suggestion Chips before first message
SUGGESTIONS = {
    ":material/image_search: Explain attached image": "Please describe this image in detail and highlight key objects or text present.",
    ":material/short_text: Extract text (OCR)": "Please extract all readable text from the attached image.",
    ":material/help_center: What can you do?": "What visual and text capabilities do you offer?",
    ":material/code: Analyze technical layout": "Please provide a technical analysis of this image or diagram."
}

if not st.session_state["messages"]:
    selected_chip = st.pills(
        "Starter Prompts:", 
        options=list(SUGGESTIONS.keys()), 
        label_visibility="visible"
    )
    if selected_chip:
        initial_prompt = SUGGESTIONS[selected_chip]
        # Auto trigger selected suggestion
        st.session_state["pending_prompt"] = initial_prompt
        st.rerun()

# Check for pending prompt from pills
prompt_input = None
if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
    prompt_input = st.session_state.pop("pending_prompt")

# Render Existing Chat Messages
for msg in st.session_state["messages"]:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("image") is not None:
            st.image(msg["image"], width="stretch")
        st.markdown(msg["content"])

# Chat Input Widget (supports direct image attachment if needed)
chat_submission = st.chat_input(
    "Ask a question or upload an image...",
    accept_file=True,
    file_type=["png", "jpg", "jpeg", "webp"]
)

# Process User Submission
current_prompt = None
attached_image = None

if prompt_input:
    current_prompt = prompt_input
    attached_image = st.session_state["active_image"]
elif chat_submission:
    # Check if text was submitted
    if hasattr(chat_submission, "text") and chat_submission.text:
        current_prompt = chat_submission.text
    elif isinstance(chat_submission, str):
        current_prompt = chat_submission

    # Check if a file was attached directly in chat_input
    if hasattr(chat_submission, "files") and chat_submission.files:
        try:
            file_obj = chat_submission.files[0]
            img = Image.open(file_obj)
            st.session_state["active_image"] = img
            st.session_state["image_name"] = file_obj.name
            attached_image = img
        except Exception as e:
            st.error(f"Could not read attached file: {e}")
    else:
        attached_image = st.session_state["active_image"]

# Execute Gemini Request & Stream Response
if current_prompt:
    if not api_key:
        st.error("⚠️ Gemini API Key is missing. Please add it to your `dotenv` / `.env` file or enter it in the sidebar.")
        st.stop()

    # Append User Message to History
    user_msg = {
        "role": "user",
        "content": current_prompt,
        "image": attached_image
    }
    st.session_state["messages"].append(user_msg)

    # Render User Turn immediately
    with st.chat_message("user", avatar=":material/person:"):
        if attached_image:
            st.image(attached_image, width="stretch")
        st.markdown(current_prompt)

    # Render Assistant Turn & Stream Response
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        service = GeminiService(api_key=api_key, model_name=selected_model)
        
        response_stream = service.stream_chat(
            history=st.session_state["messages"][:-1],
            user_text=current_prompt,
            image=attached_image,
            system_instruction=system_instruction
        )
        
        full_response = st.write_stream(response_stream)

    # Append Assistant Message to History
    st.session_state["messages"].append({
        "role": "assistant",
        "content": full_response or ""
    })
