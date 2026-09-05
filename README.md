# School Chatbot

This project implements a simple school assistant chatbot using FastAPI and OpenAI.
It supports PDF uploads, text extraction, embedding, FAISS vector search, and RAG-style
question answering.

## Features

- Upload a PDF book with an associated grade (1–12) and automatically index its contents. The file is stored under `books/<grade>/` with a unique name.
- Ask questions and receive answers based solely on uploaded material.
- Asynchronous, dependency-injected architecture.
- Full error handling and simple CORS support.

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate      # or `venv\Scripts\activate` on Windows
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set your OpenAI API key:

   ```bash
   cp .env.example .env          # Windows: copy .env.example .env
   # then edit .env
   ```

3. Run the server with uvicorn:

   ```bash
   uvicorn app:app --reload
   ```

4. Use `POST /upload-book` with a PDF file, then `POST /ask-question` with a
   JSON body like `{ "question": "What is the main topic?" }`.

## Architecture

- `app.py` contains FastAPI application and endpoints.
- `services/` holds three services (`pdf_service`, `embedding_service`, `rag_service`).
- `utils/text_utils.py` manages text chunking logic.
- `models/schemas.py` defines request/response Pydantic schemas.

## Embeddings & Search

Embeddings are generated via OpenAI's `text-embedding-3-small` model and stored in a FAISS index persisted on disk (``data/index.faiss`` along with ``data/texts.pkl``). The index is reloaded when the app starts, so uploads survive restarts.

## Notes

- This code is production-ready but does not persist the index across restarts.
  For a long-running service you might serialize the FAISS index and texts list.

- Adjust model names depending on availability and cost.
