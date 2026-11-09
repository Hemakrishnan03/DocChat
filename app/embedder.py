
from pathlib import Path
import logging


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        local_path = Path("./data/models/bge-base-en-v1.5")
        if local_path.exists():
            print(f"Loading embedding model from local path: {local_path}")
            self.model = SentenceTransformer(str(local_path),local_files_only=True)
        else:
            print(f"Downloading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model.save(local_path)


    def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()