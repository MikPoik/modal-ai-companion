# embedding_api.py
from typing import List, Dict
from src.common import AgentConfig

class EmbeddingAPI:
    def __init__(self, api_config: AgentConfig):
        from openai import OpenAI
        import os
        self.base_url = "https://api.deepinfra.com/v1/openai"
        self.config = api_config
        self.client = OpenAI(base_url=self.base_url,api_key= os.environ["DEEP_INFRA_API_KEY"])
        self.model = "sentence-transformers/all-MiniLM-L6-v2"  # Default model, can be made configurable

    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self.client.embeddings.create(input=[text], model=self.model, encoding_format="float").data[0].embedding

    def generate_embedding(self, text: str) -> List[float]:
        return self.get_embedding(text)

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        print("Generating embeddings...")
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model, encoding_format="float")
        return [item.embedding for item in response.data]

    def chunk_text(self, text: str, chunk_size: int = 50) -> List[str]:
        words = text.split()
        return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def embed_long_text(self, text: str, chunk_size: int = 100) -> List[Dict[str, List[float]]]:
        print("Chunk and embed")
        chunks = self.chunk_text(text, chunk_size)
        embeddings = self.generate_embeddings(chunks)
        return [{"chunk": chunk, "embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]