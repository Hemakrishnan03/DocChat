

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

class LoadFile:
    def __init__(self, file_path: str, embedder, collection):
        self.file_path = file_path
        self.embedder = embedder
        self.collection = collection

    def load(self) -> bool:
        # Implement the logic to load the file and return its content
        try:
            pdfLoader = PyPDFLoader(self.file_path)
            documents = pdfLoader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_documents(documents)
            for i, text in enumerate(texts):
                # Generate embeddings for each document
                embedding = self.embedder.embed(text.page_content)
                self.collection.upsert(
                    documents=[text.page_content],
                    metadatas=[{"source": self.file_path, "page": i}],
                    ids=[f"{self.file_path}_page_{i}"],
                    embeddings=[embedding]
                )
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False

    