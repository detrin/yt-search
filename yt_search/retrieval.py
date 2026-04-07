import pickle

import numpy as np

from yt_search.models import embed, rerank

RRF_K = 60


def _rrf(ranks_list, n):
    scores = np.zeros(n)
    for ranks in ranks_list:
        for rank, idx in enumerate(ranks):
            scores[idx] += 1.0 / (RRF_K + rank + 1)
    return scores


def load(session_path):
    import faiss
    index = faiss.read_index(str(session_path / "index.faiss"))
    with open(session_path / "data.pkl", "rb") as f:
        data = pickle.load(f)
    return index, data["bm25"], data["chunks"]


def search(query, index, bm25, chunks, top_k=20, top_n=5):
    n = len(chunks)
    top_k = min(top_k, n)

    q_emb = embed().encode(
        [f"search_query: {query}"], normalize_embeddings=True
    ).astype(np.float32)
    _, dense_ids = index.search(q_emb, top_k)
    dense_ranks = dense_ids[0].tolist()

    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_ranks = np.argsort(bm25_scores)[::-1][:top_k].tolist()

    rrf_scores = _rrf([dense_ranks, sparse_ranks], n)
    top_ids = np.argsort(rrf_scores)[::-1][:top_k].tolist()
    candidates = [chunks[i] for i in top_ids]

    try:
        scores = rerank().predict([[query, c["raw"]] for c in candidates])
        ranked = sorted(zip(scores, candidates), reverse=True)
    except Exception:
        # Reranker unavailable (model not cached) — fall back to RRF scores
        rrf_top = rrf_scores[top_ids]
        ranked = sorted(zip(rrf_top.tolist(), candidates), key=lambda x: x[0], reverse=True)

    return [{**c, "score": float(s)} for s, c in ranked[:top_n]]
