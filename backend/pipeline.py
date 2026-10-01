# pipeline.py
# Wires dense retrieval, BM25, RRF fusion, and Cross-Encoder reranking.

from backend.embedding import SentenceTransformerEmbedder
from backend.vector_store import VectorStore
from backend.sparse_retriever import SparseRetriever
from backend.fusion import reciprocal_rank_fusion
from backend.reranker import rerank

_embedder = SentenceTransformerEmbedder()
_vector_store = VectorStore()
_sparse_retriever = SparseRetriever()


def vector_only(
    query: str,
    top_k: int = 5,
) -> list[dict]:

    query_vector = _embedder([query])[0]

    return _vector_store.query(
        query_vector,
        top_k=top_k,
    )


def bm25_only(
    query: str,
    top_k: int = 5,
) -> list[dict]:

    return _sparse_retriever.query(
        query,
        top_k=top_k,
    )


def hybrid(
    query: str,
    top_k: int = 5,
    candidate_k: int = 10,
) -> list[dict]:

    query_vector = _embedder([query])[0]

    dense_results = _vector_store.query(
        query_vector,
        top_k=candidate_k,
    )

    sparse_results = _sparse_retriever.query(
        query,
        top_k=candidate_k,
    )

    return reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        top_k=top_k,
    )


def hybrid_reranked(
    query: str,
    top_k: int = 5,
    candidate_k: int = 20,
) -> list[dict]:

    fused_results = hybrid(
        query,
        top_k=candidate_k,
        candidate_k=candidate_k,
    )

    return rerank(
        query,
        fused_results,
        top_k=top_k,
    )
