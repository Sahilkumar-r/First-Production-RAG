# Production RAG Pipeline

An event-driven Retrieval-Augmented Generation (RAG) application designed to process and query document embeddings within free-tier cloud constraints.

## Tech Stack
* **Backend:** FastAPI
* **Orchestration:** Inngest
* **Vector Database:** Qdrant Cloud
* **Embeddings:** Google GenAI (`gemini-embedding-2`)
* **LLM:** Groq (`openai/gpt-oss-20b`)
* **Frontend:** Streamlit

## Architecture
* Asynchronous PDF ingestion and text chunking.
* Stateless design utilizing cloud-managed APIs for vector storage and inference.
* Resilient workflow execution with automatic retries and rate-limiting via Inngest.
