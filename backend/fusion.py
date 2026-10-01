# fusion.py
# Combines dense vector search and BM25 search using
# Reciprocal Rank Fusion (RRF).

def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
    top_k: int = 10,
) -> list[dict]:

    scores = {}
    payloads = {}

    for rank, result in enumerate(dense_results, start=1):
        chunk_id = result["id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        payloads[chunk_id] = result

    for rank, result in enumerate(sparse_results, start=1):
        chunk_id = result["id"]

        scores[chunk_id] = (
            scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

        payloads.setdefault(chunk_id, result)

    ranked_ids = sorted(
        scores,
        key=lambda cid: scores[cid],
        reverse=True,
    )[:top_k]

    return [
        {
            **payloads[cid],
            "rrf_score": scores[cid],
        }
        for cid in ranked_ids
    ]