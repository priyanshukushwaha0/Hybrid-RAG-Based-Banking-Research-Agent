from backend.chat_model import GroqChatModel
from backend.pipeline import hybrid_reranked

RAG_SYSTEM_PROMPT = """
You are a Banking Research Agent.
Use ONLY the supplied annual-report context for report-based financial facts.
Never invent financial numbers. Cite factual claims with source filename and page.
If the context does not contain the answer, say that the supplied reports do not contain enough information.
Do not provide personalized investment advice.
"""

CHAT_SYSTEM_PROMPT = """
You are a helpful general-purpose AI assistant.
Have a natural conversation with the user and answer clearly.
Do not invent specific financial/report figures when no source context is available.
"""

RAG_PROMPT = """
Retrieved annual-report context:
{context}

User question:
{query}

Answer the question using only the context.
Include a Sources section. For every report-based factual claim, include source filename and page.

Answer:
"""


def format_context(chunks):
    return "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )


def banking_answer(query: str, top_k: int = 5):
    chunks = hybrid_reranked(
        query,
        top_k=top_k,
        candidate_k=max(20, top_k * 4),
    )

    if not chunks:
        return {
            "answer": "I could not find relevant information in the ingested annual reports.",
            "sources": [],
            "mode": "banking_rag",
        }

    prompt = RAG_PROMPT.format(
        context=format_context(chunks),
        query=query,
    )

    answer = GroqChatModel()(RAG_SYSTEM_PROMPT, prompt)

    sources = [
        {
            "source": c["source"],
            "page": c["page"],
            "rerank_score": c.get("rerank_score"),
        }
        for c in chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "mode": "banking_rag",
    }


def general_chat(query: str):
    answer = GroqChatModel()(CHAT_SYSTEM_PROMPT, query)
    return {
        "answer": answer,
        "sources": [],
        "mode": "normal_chat",
    }