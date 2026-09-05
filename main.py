import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Body, HTTPException
import inngest
from groq import Groq
from data_loader import embed_texts  # Assumes embed_texts is defined here or imported correctly
from vector_db import QdrantStorage

app = FastAPI(title="Production RAG Backend")

# Initialize Inngest client
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

@app.post("/api/query")
async def api_query(data: dict = Body(...)):
    question = data.get("question")
    top_k = data.get("top_k", 5)
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
        
    # 1. Convert question to vector embedding
    query_vector = embed_texts([question])[0]
    
    # 2. Search Qdrant storage
    storage = QdrantStorage(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    search_results = storage.search(query_vector, top_k=top_k)
    
    # 3. Safely format context text whether results are dicts, objects, or strings
    context_chunks = []
    for r in search_results:
        if isinstance(r, dict):
            context_chunks.append(r.get("text", ""))
        elif hasattr(r, "text"):
            context_chunks.append(r.text)
        else:
            context_chunks.append(str(r))
            
    context_text = "\n\n".join(context_chunks)
    
    # 4. Generate answer using Groq LLM
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
