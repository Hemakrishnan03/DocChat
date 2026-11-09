import os
from typing import Any

import chromadb
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI,Request
from contextlib import asynccontextmanager
from app.generator import Generator
from app.loadFile import LoadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Starting RAG application")
    from app.loadModel import ModelLoader
    app.state.model_loader = ModelLoader()
    app.state.chroma_client = chromadb.PersistentClient(
        path=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    )
    app.state.pdf_collection = app.state.chroma_client.get_or_create_collection(
        name="pdf_data"
    )
    yield
    # Shutdown logic
    del app.state.model_loader

app = FastAPI(
    title=os.getenv("APP_NAME", "RAG Application"),
    lifespan=lifespan,
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "FastAPI service is running"}

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/load/pdfdata")
def load_pdf_data(request: Request) -> dict[str, str]:
    # Load the PDF data here
    embedder = request.app.state.model_loader.embedder
    file_loader = LoadFile(
        "./data/Refund_Process.pdf",
        embedder,
        request.app.state.pdf_collection,
    )
    if file_loader.load():
     return {"message": "PDF data loaded successfully"}
    else:
     return {"message": "Failed to load PDF data"}


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query")
def query_data(body: QueryRequest, request: Request) -> dict[str, Any]:
    generator = Generator(
        request.app.state.model_loader,
        request.app.state.pdf_collection,
      )
    return generator.query(body.query, body.top_k)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )