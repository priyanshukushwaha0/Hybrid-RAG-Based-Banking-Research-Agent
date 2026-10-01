# embedding.py
# Creates dense embeddings for ingestion and query-time retrieval.
# The same local SentenceTransformer model is used in both places.

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

_EMBED_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

BATCH_LIMIT = 64


class SentenceTransformerEmbedder:
    # Single embedding wrapper so ingestion and query-time embedding
    # always use the same model.

    def __init__(self):
        self.model = SentenceTransformer(_EMBED_MODEL)

    def __call__(self, texts: list[str], on_progress=None) -> list[list[float]]:
        if not texts:
            return []

        all_vectors = []
        total_batches = (len(texts) + BATCH_LIMIT - 1) // BATCH_LIMIT

        for batch_num, start in enumerate(
            range(0, len(texts), BATCH_LIMIT),
            start=1,
        ):
            batch = texts[start:start + BATCH_LIMIT]

            vectors = self.model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            all_vectors.extend(vectors.tolist())

            if on_progress:
                on_progress(batch_num, total_batches)

        return all_vectors

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
