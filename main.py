import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
import inngest
import inngest.fast_api
from groq import Groq
from data_loader import embed_texts  # Assumes embed_texts is defined here or imported correctly[cite: 1]
from vector_db import QdrantStorage

app = FastAPI(title="Production RAG Backend")

# Initialize Inngest client[cite: 1]
inngest_client = inngest.Inngest(
    app_id="first_production_rag",
    is_production=True,
    event_key=os.getenv("INNGEST_EVENT_KEY"),
)

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

@inngest_client.create_function(
    fn_id="ingest-pdf-background",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def ingest_pdf_background(ctx: inngest.Context, step: inngest.Step):
    data = ctx.event.data
    pdf_path = data.get("pdf_path")
    source_id = data.get("source_id")
    
    def process_pdf():
        from data_loader import load_pdf_chunks, embed_texts
        from vector_db import QdrantStorage
        
        chunks = load_pdf_chunks(pdf_path)
        if not chunks:
            return 0
            
        texts = [chunk.get("text") if isinstance(chunk, dict) else str(chunk) for chunk in chunks]
        embeddings = embed_texts(texts)
        
        storage = QdrantStorage(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        storage.upsert(
            texts=texts,
            embeddings=embeddings,
            source_id=source_id
        )
        return len(texts)

    chunk_count = await step.run("process-and-store-pdf", process_pdf)
    return {"status": "processed", "source": source_id, "chunks_indexed": chunk_count}

@app.post("/api/query")
async def api_query(data: dict = Body(...)):
    question = data.get("question")
    top_k = data.get("top_k", 5)
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
        
    # 1. Convert question to vector embedding[cite: 1]
    query_vector = embed_texts([question])[0]
    
    # 2. Search Qdrant storage[cite: 1]
    storage = QdrantStorage(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    search_results = storage.search(query_vector, top_k=top_k)
    
    # 3. Safely format context text whether results are dicts, objects, or strings[cite: 1]
    context_chunks = []
    for r in search_results:
        if isinstance(r, dict):
            context_chunks.append(r.get("text", ""))
        elif hasattr(r, "text"):
            context_chunks.append(r.text)
        else:
            context_chunks.append(str(r))
            
    context_text = "\n\n".join(context_chunks)
    
    # 4. Generate answer using Groq LLM[cite: 1]
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": "Answer the question based strictly on the provided context."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"}
        ]
    )
    answer = completion.choices[0].message.content
    return {"status": "success", "answer": answer, "sources": search_results}

# Expose Inngest serve endpoint so Inngest Cloud can trigger background tasks
inngest.fast_api.serve(
    app,
    inngest_client,
    [ingest_pdf_background],
)
