# ingest.py
# Reads bank annual-report PDFs from data/.
# LangChain performs document loading and recursive chunking.
# Embeddings are saved to Qdrant.
# BM25 is built over the same chunks.

import os
import uuid
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.embedding import SentenceTransformerEmbedder
from backend.vector_store import VectorStore
from backend.sparse_retriever import SparseRetriever

load_dotenv()

DATA_DIR = "data"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def load_chunks() -> list[dict]:
    records = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(
            f"{DATA_DIR}/ directory does not exist."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    for filename in sorted(os.listdir(DATA_DIR)):

        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(
            DATA_DIR,
            filename,
        )

        print(f"Loading: {filename}")

        loader = PyPDFLoader(path)
        documents = loader.load()

        chunks = splitter.split_documents(
            documents
        )

        for chunk in chunks:
            text = chunk.page_content.strip()

            if not text:
                continue

            page_number = int(
                chunk.metadata.get("page", 0)
            ) + 1

            records.append(
                {
                    "text": text,
                    "source": filename,
                    "page": page_number,
                }
            )

    return records


def print_embedding_progress(
    batch_num: int,
    total_batches: int,
):
    percent = int(
        batch_num / total_batches * 100
    )

    print(
        f"\rEmbedding chunks: "
        f"{percent}% ({batch_num}/{total_batches} batches)",
        end="",
        flush=True,
    )


def print_upsert_progress(
    batch_num: int,
    total_batches: int,
):
    percent = int(
        batch_num / total_batches * 100
    )

    print(
        f"\rSaving to Qdrant: "
        f"{percent}% ({batch_num}/{total_batches} batches)",
        end="",
        flush=True,
    )


def run():
    records = load_chunks()

    print(
        f"Loaded {len(records)} chunks "
        f"from {DATA_DIR}/"
    )

    if not records:
        raise RuntimeError(
            "No PDF chunks found. Put annual-report PDFs in data/."
        )

    texts = [
        record["text"]
        for record in records
    ]

    embedder = SentenceTransformerEmbedder()

    vectors = embedder(
        texts,
        on_progress=print_embedding_progress,
    )

    print()

    store = VectorStore()

    store.ensure_collection(
        vector_size=len(vectors[0])
    )

    ids = [
        str(uuid.uuid4())
        for _ in records
    ]

    payloads = [
        {
            "text": record["text"],
            "source": record["source"],
            "page": record["page"],
        }
        for record in records
    ]

    store.upsert(
        ids=ids,
        vectors=vectors,
        payloads=payloads,
        on_progress=print_upsert_progress,
    )

    print()

    print(
        f"Saved {len(records)} chunks "
        f"to Qdrant"
    )

    sparse = SparseRetriever()

    sparse.build_index(
        ids=ids,
        texts=texts,
        payloads=payloads,
    )

    print(
        f"Built BM25 index over "
        f"{len(records)} chunks"
    )


if __name__ == "__main__":
    run()