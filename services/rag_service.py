import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from services.embedding_service import EmbeddingService


# new client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not configured; set it in .env or environment")
client = OpenAI(api_key=api_key)

CHAT_MODEL = "gpt-4o-mini"  # adjust as needed for cost/performance


class RAGService:
    """Orchestrates retrieval and generation to answer user questions."""

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service

    def answer(self, question: str) -> str:
        # 1. retrieve relevant chunks
        chunks: List[str] = self.embedding_service.search(question, top_k=3)
        if not chunks:
            return "I'm sorry, I could not find information in the uploaded book."

        # 2. build a simple prompt that instructs the model to use only provided text
        prompt = (
            "Answer only using the provided book content.\n\n"
            "Book excerpts:\n"
            f"{chr(10).join(chunks)}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 3. call the chat model
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        # response format: .choices is list of objects with .message.content
        text = response.choices[0].message.content.strip()
        return text
