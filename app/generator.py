class Generator:
     def __init__(self, model_loader, collection):
        self.model_loader = model_loader
        self.collection = collection

     def query(self, text: str, top_k: int = 3) -> dict:
        embedding = self.model_loader.embedder.embed(text)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return self.prompt(results, text)

     def prompt(self, results:dict, text:str) -> dict[str, str]:
         context = "\n".join([doc for doc in results["documents"][0]])
         prompt = f"""
            You are a helpful assistant. Use the following context to answer the question.
            Context: {context}
            Question: {text}
            Answer: """
         output = self.model_loader.llm.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.7,
         )
         # print(f"Prompt: {prompt}")
         # print(f"Output: {output}")
         return {"answer": output}

