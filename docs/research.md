# Research & Design Notes

## Overview

This document captures findings from a literature review on SOTA embedding models, RAG pipeline design, and the proposed agentic architecture for yt-search.

---

## Proposed Architecture

The project is being redesigned from a user-facing multi-service app (Gradio + FastAPI + Redis + Ollama + Docker Compose) into a lightweight CLI tool consumed by an AI agent (Claude Code). The agent becomes the reasoning layer; the tool handles retrieval only.

```
yt-search download <url|query>          # build ephemeral index → session-id
yt-search query <session-id> "<text>"   # → JSON [{chunk, video, timestamp, score}]
yt-search list                          # → active sessions + video titles
yt-search clear [session-id|--all]      # manual cleanup
```

Removed: Gradio, FastAPI, Redis, Ollama QA step, Docker Compose, HTTP auth, async job queue.

The synthesis step previously handled by `llama3.1:8b` is delegated to Claude, which handles long-context reasoning significantly better.

---

## SOTA Embedding Models

Embedding model selection is benchmarked on MTEB (Massive Text Embedding Benchmark) [1], recently expanded to MMTEB for multilingual coverage [2].

### Candidates

| Model | Size | Context | License | Notes |
|---|---|---|---|---|
| `all-mpnet-base-v2` *(current)* | 420MB | 384 tokens | Apache 2.0 | Strong baseline, released 2021 |
| `nomic-embed-text-v1.5` | 137MB | 8192 tokens | Apache 2.0 | Best size/quality tradeoff; long context critical for transcript chunks |
| `BAAI/bge-m3` | 570MB | 8192 tokens | MIT | Unified dense + sparse + ColBERT in one model [3]; can replace both FAISS and BM25 |
| `BAAI/bge-en-icl` | ~7B | 32768 tokens | MIT | Few-shot LLM-based embedder [4]; best quality, overkill for local use |

### Recommendation

**`nomic-embed-text-v1.5`** for the primary swap: 3× smaller than the current model, 21× longer context window (8192 vs 384 tokens), stronger MTEB retrieval scores. The 384-token context of `all-mpnet-base-v2` causes silent truncation on transcript chunks — a critical flaw.

**`BAAI/bge-m3`** as an alternative if BM25 elimination is desired: its sparse retrieval head replaces BM25 while the dense head replaces FAISS, unifying the two retrievers into one model with a single inference pass [3].

---

## SOTA RAG Pipeline

Based on an empirical benchmark of RAG component combinations [5], the following configurations outperform the current implementation.

### Chunking

**Current:** Fixed 500-character chunks, 100-character overlap.

**Problems:**
- Character-based size ignores tokenization — a 500-char chunk may be 80 or 200 tokens depending on content, causing inconsistent embedding quality.
- Fixed boundaries split mid-sentence, losing semantic coherence.
- No metadata in the chunk text, so the embedding has no context about which video or when.

**SOTA approach:**

1. **Sentence-level chunking** beats fixed-char for short factual retrieval [5]. Split at sentence boundaries, then group into chunks of ~256–512 *tokens* (not chars).

2. **Scene-aware chunking** — segment at topic/speaker-turn boundaries using ASR timestamps [6]. For YouTube transcripts this means respecting natural pause markers in the SRT format rather than ignoring them.

3. **Metadata prepending** [5] — prepend a header to each chunk *before* embedding:
   ```
   [Video: {title} | Time: {timestamp}]
   {chunk text}
   ```
   This is a free retrieval boost: the embedding encodes source context alongside content, improving cross-video disambiguation.

### Retrieval

**Current:** Ensemble of BM25 (70%) + FAISS dense (30%), fixed weights.

**Problems:**
- Fixed weights are suboptimal — no evidence the 70:30 ratio is correct for this domain.
- KohakuRAG [7] found dense retrieval alone nearly matches hybrid BM25+dense on many tasks (BM25 adds only +3.1pp), suggesting over-reliance on BM25 here.

**SOTA approach:**

1. **Reciprocal Rank Fusion (RRF)** [5] replaces fixed-weight ensemble:
   ```
   score(d) = Σ 1 / (k + rank_i(d)),  k=60
   ```
   RRF is parameter-free, consistently outperforms fixed linear weighting, and requires no tuning.

2. **HyDE (Hypothetical Document Embedding)** [5] — before retrieval, prompt the LLM to generate a hypothetical "ideal answer" passage, then embed *that* for the vector search instead of the raw query. Bridges the query-document vocabulary gap. Strong win on abstract or conceptual queries ("what does X think about Y").

### Reranking

**Current:** No reranking step.

Cross-encoder reranking on top of hybrid retrieval is the current SOTA pattern [8]. The retriever fetches top-20 candidates; a cross-encoder scores each query-chunk pair jointly (not independently), then the top-5 are passed to the LLM.

**Recommended model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — 6MB, fast CPU inference, significant precision improvement.

Reranking at depth 20→5 is the sweet spot per ablation studies in [8].

### Document Repacking

After reranking, chunk ordering in the context matters. LLMs attend more strongly to text at the start and end of context than the middle ("lost in the middle" effect). **Sides packing** [5]: place most relevant chunks at the beginning and end, least relevant in the middle. Free quality improvement at zero compute cost.

---

## Session Management

### The Problem

A stateful index is needed for multi-query sessions (Claude making several queries against the same video set), but manual deletion creates UX friction.

### Options Considered

**Option A — TTL-based expiry:** Each session directory carries a `meta.json` with creation timestamp. Sessions older than N hours are purged on next invocation. Simple but does not reuse an index if Claude Code restarts mid-session.

**Option B — Process-scoped (PPID):** Session dir keyed to `PPID`. Dies when the parent process exits. Fragile due to PID reuse; not recommended.

**Option C — Content-addressed caching with TTL guard (recommended):**

Session key = `sha256(sorted(video_ids))[:8]`. Same video set → same key → reuse existing index, even across Claude Code restarts.

```
~/.cache/yt-search/
  a3f2c1b4/
    meta.json     ← {"created": 1712345678, "videos": ["dQw4w9WgXcQ", ...], "ttl_hours": 24}
    index.faiss
    chunks.pkl
    bm25.pkl
```

Properties:
- **Zero friction:** Claude never manually manages sessions for the same video set.
- **Cross-restart reuse:** Index persists in `~/.cache/` (not `/tmp/`), survives Claude Code restarts within the TTL window.
- **Natural deduplication:** Adding or removing a single video produces a different hash → fresh index.
- **Automatic cleanup:** TTL of 24 hours means stale sessions self-expire without any cron job — the check runs on next `yt-search` invocation.
- **Manual override:** `yt-search clear [session-id|--all]` for explicit cleanup.

This design means the common case (Claude researching the same set of videos across a multi-hour session) requires zero management, while long-abandoned indices clean themselves up automatically.

---

## References

[1] Thakur, N. et al. "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models." NeurIPS 2021. (MTEB leaderboard: huggingface.co/spaces/mteb/leaderboard)

[2] Enevoldsen, K. et al. "MMTEB: Massive Multilingual Text Embedding Benchmark." arXiv:2502.13595, Feb 2025.

[3] Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z. "M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation." arXiv:2402.03216, Feb 2024.

[4] Li, C. et al. "Making Text Embedders Few-Shot Learners." arXiv:2409.15700, Sep 2024.

[5] Wang, X. et al. "Searching for Best Practices in Retrieval-Augmented Generation." arXiv:2407.01219, Jul 2024.

[6] Zeng, N. et al. "SceneRAG: Scene-level Retrieval-Augmented Generation for Video Understanding." arXiv:2506.07600, Jun 2025.

[7] Yeh, S., Ku, Y., Huang, K., Tu, B. "KohakuRAG: A simple RAG framework with hierarchical document indexing." arXiv:2603.07612, Mar 2026.

[8] Akarsu, M., Karaman, R.K., Mierbach, C. "From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents." arXiv:2604.01733, Apr 2026.
