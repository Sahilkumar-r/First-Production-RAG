import logging
import os
import uuid
import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from groq import Groq 

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

# 1. Load the environment variables from .env
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Production RAG Backend")

# 2. Production-Ready Inngest Client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("INNGEST_DEV") is None, 
    serializer=inngest.PydanticSerializer()
)

# --- INNGEST BACKGROUND INGESTION (Your original working logic) ---
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
    ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", Path(pdf_path).name)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        
        store = QdrantStorage(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )
        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    
    return ingested.model_dump()


# --- FASTAPI ENDPOINT: RECEIVE FILE & TRIGGER INNGEST ---
@app.post("/api/trigger-ingest")
async def api_trigger_ingest(file: UploadFile = File(...)):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": str(file_path.resolve()), "source_id": file.filename}
        )
    )
    return {"status": "success", "event_id": event_ids[0] if event_ids else None}


# --- FASTAPI ENDPOINT: SYNCHRONOUS QUERY (Fixes Streamlit Timeouts) ---
@app.post("/api/query")
async def api_query(data: dict = Body(...)):
    question = data.get("question")
    top_k = int(data.get("top_k", 5))
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    # 1. Search the Vector Database
    query_vec = embed_texts([question])[0]
    store = QdrantStorage(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )
    found = store.search(query_vec, top_k)
    
    # Extract contexts and sources based on your Qdrant return structure
    contexts = found.get("contexts", [])
    sources = found.get("sources", [])
    
    # 2. Format Context
    context_block = "\n\n".join(f"- {c}" for c in contexts)

    # 3. Generate Answer via Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )
    
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b", 
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    answer = completion.choices[0].message.content.strip()

    return {"answer": answer, "sources": sources, "num_contexts": len(contexts)}


# --- REGISTER INNGEST ROUTES ---
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])
