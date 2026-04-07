# Retrieval Stack Reference

## Embedding

Model: `nomic-ai/nomic-embed-text-v1.5`
- 8192-token context window (vs 384 for the previous all-mpnet-base-v2)
- Requires task prefixes: documents get `search_document:`, queries get `search_query:`
- ~137MB, Apache 2.0

## Chunking

- ~400-word chunks at segment boundaries from SRT timestamps
- Metadata prepended to each chunk before embedding: `[Video: {title} | Time: HH:MM:SS]`
- Consecutive duplicate lines deduplicated (YouTube auto-caption artifact)

## Retrieval — Reciprocal Rank Fusion

Dense (FAISS cosine) + sparse (BM25) results merged via RRF:

```
score(d) = Σ 1 / (60 + rank_i(d))
```

Top-20 candidates selected for reranking.

## Reranking

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~6MB)
- Scores each (query, chunk) pair jointly
- Top-20 → top-5 (default), adjustable with `--top-n`

## Session storage

`~/.cache/yt-search/{sha256(sorted_video_ids)[:8]}/`
- `index.faiss` — FAISS flat inner-product index
- `data.pkl` — BM25 index + chunk list
- `meta.json` — creation time, video IDs, titles, chunk count
- TTL: 24 hours, checked on every invocation
