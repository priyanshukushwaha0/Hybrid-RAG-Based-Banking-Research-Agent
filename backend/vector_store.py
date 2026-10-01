# vector_store.py
# Handles Qdrant vector storage and similarity search.
# Qdrant stores the dense embeddings and document metadata payloads.

import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
_QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip() or None
_COLLECTION = os.environ.get(
    "QDRANT_COLLECTION",
    "bank_documents",
)


class VectorStore:
    """Qdrant-backed dense vector store."""

    def __init__(self):
        self.url = _QDRANT_URL
        self.collection = _COLLECTION

        self.client = QdrantClient(
            url=self.url,
            api_key=_QDRANT_API_KEY,
        )

    def ensure_collection(self, vector_size: int):
        """Create the Qdrant collection if it does not exist.

        If an existing collection was created with a different embedding
        dimension, fail clearly instead of silently storing incompatible data.
        """
        try:
            exists = self.client.collection_exists(self.collection)

            if not exists:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
                print(
                    f"Created Qdrant collection '{self.collection}' "
                    f"with vector size {vector_size}"
                )
                return

            info = self.client.get_collection(self.collection)
            vectors = info.config.params.vectors

            # This project uses a single unnamed dense vector.
            existing_size = getattr(vectors, "size", None)

            if existing_size is not None and existing_size != vector_size:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' already exists "
                    f"with vector size {existing_size}, but the current "
                    f"embedding model produces {vector_size}-dimensional vectors. "
                    f"Either keep the same EMBEDDING_MODEL or delete the "
                    f"collection and run ingestion again."
                )

            print(
                f"Using Qdrant collection '{self.collection}' "
                f"(vector size {existing_size})"
            )

        except Exception as exc:
            raise RuntimeError(
                "Could not connect to Qdrant or initialize the collection. "
                f"QDRANT_URL={self.url}. "
                "Make sure Qdrant is running on port 6333."
            ) from exc

    # Backward-compatible method name for the ingestion pipeline.
    def ensure_table(self, vector_size: int):
        self.ensure_collection(vector_size)

    def clear(self):
        """Delete all points from the current collection."""
        if self.client.collection_exists(self.collection):
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                ),
            )

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
        on_progress=None,
    ):
        if not ids:
            return

        total = len(ids)
        batch_limit = 100
        total_batches = (total + batch_limit - 1) // batch_limit

        for batch_num, start in enumerate(
            range(0, total, batch_limit),
            start=1,
        ):
            end = min(start + batch_limit, total)

            points = []

            for i in range(start, end):
                chunk_id = str(ids[i])

                # Qdrant accepts UUID strings as point IDs.
                # Normalize arbitrary UUID strings so ingestion remains stable.
                point_id = str(uuid.UUID(chunk_id))

                payload = {
                    "text": payloads[i]["text"],
                    "source": payloads[i]["source"],
                    "page": int(payloads[i]["page"]),
                    "chunk_id": point_id,
                }

                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=vectors[i],
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection,
                points=points,
                wait=True,
            )

            if on_progress:
                on_progress(batch_num, total_batches)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        results = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        ).points

        output = []

        for result in results:
            payload = result.payload or {}

            output.append(
                {
                    "id": str(result.id),
                    "text": payload.get("text", ""),
                    "source": payload.get("source", ""),
                    "page": int(payload.get("page", 0)),
                    "score": float(result.score),
                }
            )

        return output

    def count(self) -> int:
        if not self.client.collection_exists(self.collection):
            return 0

        result = self.client.count(
            collection_name=self.collection,
            exact=True,
        )
        return int(result.count)
