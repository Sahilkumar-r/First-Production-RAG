import logging
import os
import uuid
import datetime
from fastapi import FastAPI, Body
import inngest
import inngest.fast_api
from dotenv import load_dotenv
from groq import Groq 

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc


load_dotenv()

# 2. Production-Ready Inngest Client
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    # This automatically secures the app on Render, but stays open locally when INNGEST_DEV=1
    is_production=os.getenv("INNGEST_DEV") is None, 
    serializer=inngest.PydanticSerializer()
)

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
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        ).upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage(
            url=os.getenv("QDRANT_URL"), 
            api_key=os.getenv("QDRANT_API_KEY")
        )
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    # Extract the AI generation into its own reliable step
    def _generate_answer(context_block: str, question: str) -> str:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        user_content = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using the context above."
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b", # Free, fast Groq model
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # 1. Search the Vector Database
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # 2. Format Context
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)

    # 3. Generate Answer via Groq (Wrapped in a step for automatic retries)
    answer = await ctx.step.run("generate-llm-answer", lambda: _generate_answer(context_block, question))

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

# Initialize FastAPI and Inngest API route
app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])


# 4. Place your custom triggers last
@app.post("/api/trigger-ingest")
async def api_trigger_ingest(data: dict = Body(...)):
    async def api_trigger_ingest(file: UploadFile = File(...)):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.filename
    
    # Save the file received from your computer via Streamlit
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
        
    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={"pdf_path": str(file_path.resolve()), "source_id": file.filename}
        )
    )
    return {"status": "success", "event_id": event_ids[0] if event_ids else None}

@app.post("/api/trigger-query")
async def api_trigger_query(data: dict = Body(...)):
    question = data.get("question")
    top_k = data.get("top_k", 5)
    event_ids = await inngest_client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k}
        )
    )
    return {"status": "success", "event_id": event_ids[0] if event_ids else None}
