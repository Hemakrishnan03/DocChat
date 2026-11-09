# Chat With Your Docs

## 1. Quick setup

### Python API

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m app.main
```

The service is available at `http://127.0.0.1:8000`; 


## Angular App

```bash
cd DocChat
npm install
npm start
```

Open http://localhost:4200/

## 2. Architecture Overview

### Current Prototype

```text
Angular UI
   |
   | POST /query { query, top_k }
   v
FastAPI
   |
   +--> SentenceTransformers: embed the query
   |
   +--> ChromaDB: retrieve the top-k document chunks
   |
   +--> Transformers: generate an answer from the question and retrieved context
   |
   v
Answer returned to Angular
```

Document ingestion is a separate API operation:

```text
Local PDF
   |
   +--> PyPDFLoader
   +--> RecursiveCharacterTextSplitter (1,000 characters, 200 overlap)
   +--> BGE embeddings
   +--> ChromaDB collection "pdf_data"
```

FastAPI loads the embedding and language models during application startup. ChromaDB uses a local persistent directory, `data/chroma` by default. Angular currently sends queries to `http://127.0.0.1:8000/query`.

### Main Components

| Component | Current responsibility |
|---|---|
| Angular app | Accepts a question, submits it to the API, and displays the returned answer. |
| FastAPI | Initializes shared model and Chroma instances and exposes health, ingestion, and query endpoints. |
| `LoadFile` | Loads a PDF, splits its text, embeds chunks, and upserts them into ChromaDB. |
| `Embedder` | Loads `BAAI/bge-base-en-v1.5` from a local model directory or downloads it. |
| `Generator` | Retrieves chunks and builds a context-based prompt for answer generation. |
| `LLMLoader` | Loads `Qwen/Qwen2.5-0.5B-Instruct` and generates text with Transformers. |



## 3. RAG and LLM Decisions

### Current Choices

| Area | Current implementation |
|---|---|
| Language model | Qwen2.5-0.5B-Instruct, locally loaded with Hugging Face Transformers | 
| Embedding model | BAAI BGE Base English v1.5, via SentenceTransformers | 
| Vector database | ChromaDB `PersistentClient` on local disk | 
| Orchestration | Direct Python calls between application classes; LangChain community PDF loading and text splitting | 
| Chunking | Recursive character splitting, 1,000-character chunks with 200-character overlap | 
| Retrieval | Dense vector search, default `top_k=3` | 
| Prompt and context | Retrieved chunk text is concatenated into a short instruction prompt | 
| Generation | Up to 100 new tokens with temperature 0.7 | 


### Recommended Production Baseline on GCP

These are proposed production choices, not services currently implemented in this repository.

| Area | Recommended starting choice | 
|---|---|
| Language model | Vertex AI Gemini model selected through a deployment-time configuration | 
| Embeddings | Vertex AI text embeddings, or retain BGE if self-hosting is a requirement | 
| Vector database | Vertex AI Vector Search | 
| Orchestration | Keep a small, explicit Python RAG pipeline; use LangChain for document parsing/splitting where it helps | 
| Ingestion | Background ingestion workers writing to object storage and the managed vector index | 

## 4. Productionizing and Scaling on GCP

A practical target architecture would separate the browser app, API, ingestion, and managed data services:

```text
Browser
   |
   +--> Angular static assets: Firebase Hosting or Cloud Storage + Cloud CDN
   |
   +--> HTTPS API: API Gateway or load balancer
             |
             +--> FastAPI containers on Cloud Run
             +--> Vertex AI for generation and embeddings
             +--> Vertex AI Vector Search for retrieval
             +--> Cloud Storage for source documents
             +--> Pub/Sub and ingestion workers for asynchronous indexing
```

For the first deployment, Cloud Run is a good fit for a stateless FastAPI service when model is delegated to Vertex AI.
