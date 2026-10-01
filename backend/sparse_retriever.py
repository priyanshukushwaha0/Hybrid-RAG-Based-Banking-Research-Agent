# sparse_retriever.py
# BM25 keyword search.
# The BM25 index is saved to disk so ingestion and query-time search
# can happen in separate program runs.

import os
import re
import pickle
from rank_bm25 import BM25Okapi

_INDEX_PATH = os.environ.get(
    "BM25_INDEX_PATH",
    "data/bm25_index.pkl",
)


def _tokenize(text: str) -> list[str]:
    # Lowercase and keep alphanumeric banking terms.
    return re.findall(r"[a-z0-9]+", text.lower())


class SparseRetriever:

    def __init__(self):
        self._bm25 = None
        self._ids = []
        self._payloads = []

    def build_index(
        self,
        ids: list[str],
        texts: list[str],
        payloads: list[dict],
    ):
        tokenized_corpus = [
            _tokenize(text)
            for text in texts
        ]

        self._bm25 = BM25Okapi(tokenized_corpus)
        self._ids = ids
        self._payloads = payloads

        directory = os.path.dirname(_INDEX_PATH)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(_INDEX_PATH, "wb") as f:
            pickle.dump(
                {
                    "bm25": self._bm25,
                    "ids": self._ids,
                    "payloads": self._payloads,
                },
                f,
            )

    def load_index(self):
        if not os.path.exists(_INDEX_PATH):
            raise FileNotFoundError(
                f"BM25 index not found at {_INDEX_PATH}. "
                "Run: python -m backend.ingest"
            )

        with open(_INDEX_PATH, "rb") as f:
            state = pickle.load(f)

        self._bm25 = state["bm25"]
        self._ids = state["ids"]
        self._payloads = state["payloads"]

    def query(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[dict]:

        if self._bm25 is None:
            self.load_index()

        scores = self._bm25.get_scores(
            _tokenize(query_text)
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            {
                "id": self._ids[i],
                **self._payloads[i],
                "score": float(scores[i]),
            }
            for i in ranked_indices
        ]
