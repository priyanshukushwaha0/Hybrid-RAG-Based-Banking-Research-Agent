# reranker.py
# Cross-Encoder reranker.
# Unlike the sample's LLM reranker, this uses a dedicated
# Cross-Encoder model to score (query, document) pairs.

import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

load_dotenv()

_RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)


class CrossEncoderReranker:

    def __init__(self):
        self.model = CrossEncoder(_RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> list[dict]:

        if not chunks:
            return []

        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        scored = [
            (chunk, float(score))
            for chunk, score in zip(chunks, scores)
        ]

        scored.sort(
            key=lambda pair: pair[1],
            reverse=True,
        )

        results = []

        for chunk, score in scored[:top_k]:
            results.append(
                {
                    **chunk,
                    "rerank_score": score,
                }
            )

        return results


_reranker = None


def rerank(
    query: str,
    chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:

    global _reranker

    if _reranker is None:
        _reranker = CrossEncoderReranker()

    return _reranker.rerank(
        query,
        chunks,
        top_k=top_k,
    )
