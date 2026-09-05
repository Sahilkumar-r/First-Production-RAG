import asyncio
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

def save_uploaded_pdf(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path

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

st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

# In your file uploader section:
if uploaded is not None:
    with st.spinner("Uploading file from your computer to Render backend..."):
        try:
            trigger_backend_ingest(uploaded)
            st.success(f"Successfully uploaded and triggered ingestion for: {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to trigger ingestion: {e}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")

def query_backend(question: str, top_k: int = 5) -> dict:
    resp = requests.post(
        f"{BACKEND_URL}/api/query",
        json={"question": question, "top_k": top_k},
        timeout=60
    )
    resp.raise_for_status()
    return resp.json()

# In your Streamlit UI query section:
if st.button("Ask"):
    if user_question:
        with st.spinner("Generating answer..."):
            try:
                result = query_backend(user_question, top_k=top_k)
                st.write("### Answer")
                st.write(result.get("answer"))
            except Exception as e:
                st.error(f"Error processing query: {e}")

def fetch_runs(event_id: str) -> list[dict]:
    # In production, poll Inngest Cloud API or route through backend if preferred. 
    # Here we query Inngest Cloud Event API directly since the frontend has an event key.
    event_key = st.secrets.get("INNGEST_EVENT_KEY", os.getenv("INNGEST_EVENT_KEY", ""))
    headers = {"Authorization": f"Bearer {event_key}"} if event_key else {}
    
    # Inngest Cloud API endpoint for run tracking
    url = f"https://api.inngest.com/v1/events/{event_id}/runs"
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data.get("data", [])

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)

with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        try:
            with st.spinner("Dispatching query to backend and generating response..."):
                event_id = trigger_backend_query(question.strip(), int(top_k))
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])

            st.subheader("Answer")
            st.write(answer or "(No answer)")
            if sources:
                st.caption("Sources")
                for s in sources:
                    st.write(f"- {s}")
        except Exception as e:
            st.error(f"Error processing query: {e}")
