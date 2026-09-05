import os
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from services.pdf_service import PdfService
from services.embedding_service import EmbeddingService
from services.rag_service import RAGService

import aiofiles
import uuid
from pathlib import Path
from models.schemas import QuestionRequest, AnswerResponse


# load .env early so that the OpenAI key is available
load_dotenv()

app = FastAPI(
    title="School Chatbot",
    description="A simple RAG-style chatbot that ingests PDFs and answers questions.",
)

# CORS middleware for development/testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# dependency helpers that keep singletons in app.state

def get_pdf_service() -> PdfService:
    if not hasattr(app.state, "pdf_service"):
        app.state.pdf_service = PdfService()
    return app.state.pdf_service


def get_embedding_service() -> EmbeddingService:
    if not hasattr(app.state, "embedding_service"):
        app.state.embedding_service = EmbeddingService()
    return app.state.embedding_service


def get_rag_service(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RAGService:
    return RAGService(embedding_service)


# endpoints

@app.post("/upload-book")
async def upload_book(
    file: UploadFile = File(...),
    grade: str = Form(...),
    pdf_service: PdfService = Depends(get_pdf_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    # grade is a simple string like "grade1" or "Grade 1" – no validation here
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # create grade folder inside books/
    book_dir = Path("books") / grade
    book_dir.mkdir(parents=True, exist_ok=True)

    # save the uploaded file with a unique name
    unique_name = f"{uuid.uuid4()}.pdf"
    file_path = book_dir / unique_name
    content = await file.read()
    try:
        async with aiofiles.open(file_path, "wb") as out_f:
            await out_f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {exc}")

    # extract text and index
    try:
        text = pdf_service.extract_text(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse PDF: {exc}")

    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail="PDF contained no extractable text; is it a scanned/image file?",
        )

    ids = embedding_service.index_text(text)
    # persist the index to disk after adding new chunks
    embedding_service.save()

    return {"status": "ok", "chunks_indexed": len(ids), "saved_to": str(file_path)}


@app.post("/ask-question", response_model=AnswerResponse)
async def ask_question(
    req: QuestionRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        answer = rag_service.answer(question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {exc}")

    return AnswerResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
