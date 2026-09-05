from pathlib import Path
import time
import streamlit as st
import os
import requests

st.set_page_config(page_title="Production RAG Pipeline", page_icon="📄", layout="centered")

# Get backend URL from Streamlit secrets or environment variables, falling back to Render URL
def get_backend_url() -> str:
    try:
        return st.secrets.get("BACKEND_URL", "https://first-production-rag.onrender.com").rstrip("/")
    except Exception:
        return os.getenv("BACKEND_URL", "https://first-production-rag.onrender.com").rstrip("/")

BACKEND_URL = get_backend_url()

def trigger_backend_ingest(uploaded_file) -> dict:
    # Package the file from your computer into an HTTP request
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    resp = requests.post(
        f"{BACKEND_URL}/api/trigger-ingest",
        files=files,
        timeout=30
    )
    resp.raise_for_status()
    return resp.json()

def query_backend(question: str, top_k: int = 5) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/api/query",
        json={"question": question, "top_k": top_k},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()

# --- UI: PDF Ingestion Section ---
st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading file from your computer to Render backend..."):
        try:
            trigger_backend_ingest(uploaded)
            st.success(f"Successfully uploaded and triggered ingestion for: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to trigger ingestion: {e}")
    st.caption("You can upload another PDF if you like.")

st.divider()

# --- UI: RAG Query Section ---
st.title("Ask a question about your PDFs")

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        try:
            with st.spinner("Generating answer..."):
                result = query_backend(question.strip(), int(top_k))
                answer = result.get("answer", "")
                sources = result.get("sources", [])

            st.subheader("Answer")
            st.write(answer or "(No answer)")
            
            if sources:
                st.caption("Sources")
                for s in sources:
                    # Adjust depending on how your backend structure returns sources
                    source_name = s.get("source_id", "Unknown source") if isinstance(s, dict) else s
                    st.write(f"- {source_name}")
                    
        except Exception as e:
            st.error(f"Error processing query: {e}")
