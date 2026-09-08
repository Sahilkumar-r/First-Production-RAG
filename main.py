import logging
import os
import uuid
import json
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

# --- INNGEST BACKGROUND INGESTION ---
@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(limit=2, period=datetime.timedelta(minutes=1)),
    rate_limit=inngest.RateLimit(limit=1, period=datetime.timedelta(hours=4), key="event.data.source_id"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", Path(pdf_path).name)
        chunks = load_and_chunk_pdf(pdf_path) # Now returns rich dicts
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        
        # Embed only the enhanced summary text
        texts_to_embed = [c["enhanced_content"] for c in chunks]
        vecs = embed_texts(texts_to_embed)
        
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        
        # Store full multimodal data in Qdrant Payload
        payloads = [{
            "source": source_id, 
            "text": chunk["enhanced_content"],
            "original_content": json.dumps(chunk["original_content"])
        } for chunk in chunks]
        
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


# --- FASTAPI ENDPOINT: SYNCHRONOUS QUERY ---
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
    
    # Contexts must contain the original dictionary payload stored during upsert
    contexts = found.get("contexts", [])
    sources = found.get("sources", [])
    
    # 2. Build Multimodal Prompt
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt_text = f"Based on the following documents, please answer this question: {question}\n\nCONTENT TO ANALYZE:\n"
    messages_content = [{"type": "text", "text": prompt_text}]

    for i, ctx in enumerate(contexts):
        prompt_text += f"--- Document {i+1} ---\n"
        
        # Parse payload appropriately whether your vector DB returns a string or a raw dict
        ctx_dict = json.loads(ctx) if isinstance(ctx, str) else ctx
        
        # Retrieve the original un-summarized content
        orig_data_str = ctx_dict.get("original_content", "{}")
        orig_data = json.loads(orig_data_str) if isinstance(orig_data_str, str) else orig_data_str

        raw_text = orig_data.get("raw_text", ctx_dict.get("text", ""))
        prompt_text += f"TEXT:\n{raw_text}\n\n"
        
        tables_html = orig_data.get("tables_html", [])
        if tables_html:
            prompt_text += "TABLES:\n"
            for j, table in enumerate(tables_html):
                prompt_text += f"Table {j+1}:\n{table}\n\n"
                
        # Append images dynamically to Groq payload
        images_base64 = orig_data.get("images_base64", [])
        for img in images_base64:
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img}"}
            })

    prompt_text += "\nPlease provide a clear, comprehensive answer using the text, tables, and images above.\nANSWER:"
    messages_content[0]["text"] = prompt_text

    # 3. Generate Multimodal Answer via Groq
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview", # Groq multimodal vision model
        messages=[
            {"role": "user", "content": messages_content}
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    answer = completion.choices[0].message.content.strip()

    return {"answer": answer, "sources": sources, "num_contexts": len(contexts)}

# --- REGISTER INNGEST ROUTES ---
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf])
