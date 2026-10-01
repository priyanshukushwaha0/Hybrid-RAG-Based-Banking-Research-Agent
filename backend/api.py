import traceback

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.generate import banking_answer, general_chat
from backend.vector_store import VectorStore

app = FastAPI(
    title="Hybrid RAG-Based Banking Research Agent",
    description="Bank annual-report research using BM25, Qdrant, RRF, Cross-Encoder and Groq.",
    version="2.0.0",
)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=10)
    mode: str = Field(default="auto")  # auto | banking | chat


def is_banking_query(query: str) -> bool:
    terms = [
        "annual report", "annual-report", "bank", "banking", "net profit",
        "profit", "revenue", "income", "balance sheet", "capital adequacy",
        "npa", "gross npa", "net npa", "deposit", "deposits", "advance",
        "advances", "loan", "loans", "asset", "assets", "liability",
        "liabilities", "roa", "roe", "credit", "provision", "dividend",
        "eps", "hdfc", "icici", "sbi", "axis", "kotak", "indusind",
        "bank of baroda", "punjab national", "financial year", "fy202",
        "fy203",
    ]
    q = query.lower()
    return any(term in q for term in terms)


@app.get("/")
def root():
    return {
        "project": "Hybrid RAG-Based Banking Research Agent",
        "status": "running",
        "modes": ["auto", "banking", "chat"],
    }


@app.get("/health")
def health():
    try:
        count = VectorStore().count()
        return {"status": "healthy", "documents": count}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )


@app.post("/ask")
def ask(request: AskRequest):
    try:
        query = request.query.strip()
        mode = request.mode.lower().strip()

        if mode not in {"auto", "banking", "chat"}:
            return JSONResponse(
                status_code=400,
                content={
                    "answer": "mode must be auto, banking, or chat.",
                    "sources": [],
                    "mode": "error",
                },
            )

        if mode == "banking":
            return banking_answer(query, request.top_k)

        if mode == "chat":
            return general_chat(query)

        if is_banking_query(query):
            return banking_answer(query, request.top_k)

        return general_chat(query)

    except Exception as exc:
        print("\n========== FASTAPI ERROR ==========")
        traceback.print_exc()
        print("===================================\n")

        return JSONResponse(
            status_code=500,
            content={
                "answer": "The backend encountered an error.",
                "sources": [],
                "mode": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )