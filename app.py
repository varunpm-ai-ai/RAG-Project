"""
Streamlit Application: DOTA Aerial Image Vision & RAG Knowledge Assistant.
Integrates Ultralytics Object Detection model with Category-Specific RAG Assistant.
"""

import os
import sys
import json
import logging
from pathlib import Path
from PIL import Image
import streamlit as st

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))

from src.utils.constants import DOTA_CLASSES, ID_TO_CLASS
from src.inference.predictor import DotaPredictor
from src.rag.pipeline import RagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Page Configuration
st.set_page_config(
    page_title="DOTA Vision + Knowledge Assistant",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Cached Model & Pipeline Loading
@st.cache_resource
def get_predictor():
    model_path = Path("artifacts/best_model/model.pt")
    if not model_path.exists():
        return None
    try:
        return DotaPredictor(model_path="artifacts/best_model/model.pt")
    except Exception as e:
        logging.error(f"Error loading predictor: {e}")
        return None


@st.cache_resource
def get_rag_pipeline():
    try:
        return RagPipeline(storage_dir="rag_storage")
    except Exception as e:
        logging.error(f"Error initializing RAG pipeline: {e}")
        return None


def init_session_state():
    if "detection_results" not in st.session_state:
        st.session_state.detection_results = None
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "uploaded_docs_count" not in st.session_state:
        st.session_state.uploaded_docs_count = 0


def main():
    init_session_state()

    st.title("🛸 DOTA Aerial Vision & Knowledge Assistant")
    st.caption("Detect objects in aerial images and query category-specific knowledge via RAG.")

    predictor = get_predictor()
    rag_pipeline = get_rag_pipeline()

    # Model warning banner if Phase 1 model is missing
    if predictor is None:
        st.error(
            "⚠️ **Detection model artifact not found!**\n"
            "Please complete Phase 1 model training first by running `python train.py`."
        )

    # Sidebar: Knowledge Base & Upload Management
    with st.sidebar:
        st.header("📚 Knowledge Base")

        if rag_pipeline:
            stats = rag_pipeline.get_stats()
            st.success("✓ Base Knowledge Loaded")
            col1, col2 = st.columns(2)
            col1.metric("Indexed Chunks", stats["total_chunks"])
            col2.metric("Active Sources", stats["sources_count"])
            st.info(f"Available Categories: {stats['categories_count']}")

        st.divider()
        st.subheader("➕ Add Knowledge Document")
        st.caption("Expand knowledge base by uploading custom reference documents.")

        doc_file = st.file_uploader(
            "Upload Document",
            type=["json", "txt", "pdf", "md"],
            key="doc_uploader"
        )

        doc_category = st.selectbox(
            "Target DOTA Category",
            options=DOTA_CLASSES,
            index=0,
            key="doc_cat_select"
        )

        if st.button("Index Document", type="primary", use_container_width=True):
            if doc_file and rag_pipeline:
                # Save temp file
                upload_dir = Path("uploaded_documents")
                upload_dir.mkdir(exist_ok=True)
                temp_path = upload_dir / doc_file.name

                with open(temp_path, "wb") as f:
                    f.write(doc_file.getbuffer())

                with st.spinner("Chunking & Indexing document..."):
                    count = rag_pipeline.index_document(str(temp_path), category=doc_category)
                    st.session_state.uploaded_docs_count += 1
                    st.success(f"Indexed {count} chunks for category '{doc_category}' from {doc_file.name}!")
            elif not doc_file:
                st.warning("Please upload a document file first.")

        st.divider()
        if st.button("Reset Demo Session", use_container_width=True):
            st.session_state.detection_results = None
            st.session_state.selected_category = None
            st.session_state.chat_history = []
            if rag_pipeline:
                rag_pipeline.reset_storage()
            st.rerun()

    # Main UI Tabs / Columns
    col_left, col_right = st.columns([1, 1], gap="large")

    # LEFT COLUMN: Image Detection Workflow
    with col_left:
        st.subheader("📷 Image Object Detection")

        img_file = st.file_uploader(
            "Upload Aerial Image",
            type=["jpg", "jpeg", "png", "bmp"],
            key="img_uploader"
        )

        if img_file:
            image = Image.open(img_file)
            st.image(image, caption="Uploaded Aerial Image", use_container_width=True)

            if st.button("Run Object Detection", type="primary", use_container_width=True):
                if predictor is None:
                    st.error("Cannot run detection: Model weights missing.")
                else:
                    with st.spinner("Detecting objects..."):
                        results = predictor.predict(image, conf_threshold=0.20)
                        st.session_state.detection_results = results
                        
                        detected_classes = results["detected_classes"]
                        if detected_classes:
                            st.session_state.selected_category = detected_classes[0]
                        else:
                            st.session_state.selected_category = None

        if st.session_state.detection_results:
            res = st.session_state.detection_results
            st.divider()
            st.subheader("🎯 Detection Results")

            st.image(res["annotated_image"], caption="Annotated Image with Detections", use_container_width=True)

            detections = res["detections"]
            if not detections:
                st.warning("No DOTA objects detected above confidence threshold.")
            else:
                st.success(f"Found {len(detections)} object instance(s) across {len(res['detected_classes'])} category/categories.")
                
                # Render confidence badges
                det_cols = st.columns(3)
                for idx, det in enumerate(detections):
                    with det_cols[idx % 3]:
                        st.metric(
                            label=f"#{idx+1} {det['class_name']}",
                            value=f"{det['confidence']*100:.1f}%"
                        )

                st.divider()
                st.write("**Select Detected Object Category for RAG Knowledge Search:**")
                
                detected_classes = res["detected_classes"]
                selected_cat = st.radio(
                    "Detected Categories",
                    options=detected_classes,
                    horizontal=True,
                    key="category_radio"
                )
                st.session_state.selected_category = selected_cat

    # RIGHT COLUMN: Category-Specific RAG Chat & Retrieval
    with col_right:
        st.subheader("🤖 Category-Specific RAG Assistant")

        selected_cat = st.session_state.selected_category
        if selected_cat:
            st.info(f"📍 **Active Knowledge Filter**: `{selected_cat.upper()}`")
        else:
            st.warning("No object category selected yet. Upload and detect an image, or select a category from the sidebar.")
            # Default fallback category selector if no detection has run
            selected_cat = st.selectbox("Or Select Category Manually", options=DOTA_CLASSES, index=0)
            st.session_state.selected_category = selected_cat

        st.markdown("### Ask About The Detected Object")
        default_q = f"What information is available about {selected_cat}?" if selected_cat else "What is this object?"
        user_question = st.text_input("Question:", value=default_q)

        if st.button("Ask Knowledge Base", type="primary"):
            if not user_question.strip():
                st.warning("Please enter a question.")
            elif not rag_pipeline:
                st.error("RAG Pipeline is not initialized.")
            else:
                with st.spinner("Retrieving category knowledge & generating answer..."):
                    rag_res = rag_pipeline.answer_question(
                        question=user_question,
                        category=selected_cat
                    )
                    st.session_state.chat_history.append(rag_res)

        # Display Answer & Sources
        if st.session_state.chat_history:
            latest_res = st.session_state.chat_history[-1]
            st.divider()
            st.markdown("### 📝 Answer")
            st.markdown(latest_res["answer"])

            st.divider()
            st.markdown("### 📚 Sources Used")
            sources = latest_res["sources"]
            if sources:
                for s in sources:
                    st.caption(f"• `{s}`")
            else:
                st.caption("No external sources retrieved.")

            with st.expander("🔍 View Retrieved Text Chunks"):
                for idx, chunk in enumerate(latest_res["retrieved_chunks"], 1):
                    st.markdown(f"**Chunk #{idx}** (Source: `{chunk['metadata'].get('source', 'N/A')}`, Score: `{chunk['score']}`)**")
                    st.info(chunk["text"])


if __name__ == "__main__":
    main()
