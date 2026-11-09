from pathlib import Path
import logging
from app.embedder import Embedder
import os
from app.llmloader import LLMLoader

logging.basicConfig(level=logging.INFO)

class ModelLoader:
    def __init__(self):
        self.embedder = Embedder(os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5"))
        self.llm = LLMLoader(os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"))
        print("RAG application initialized successfully")
        