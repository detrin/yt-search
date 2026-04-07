# yt-search

Search YouTube video transcripts using a local RAG pipeline exposed as a CLI tool for AI agents.

## What it does

Downloads YouTube subtitles, builds an ephemeral hybrid retrieval index, and returns ranked transcript chunks. Designed to be called by an AI agent (Claude Code) that handles the reasoning and synthesis — the tool handles retrieval only.

## How it works

```
YouTube URL → yt-dlp (SRT) → ~400-word chunks + metadata → FAISS + BM25
                                                                  ↓
query → nomic-embed-text-v1.5 + BM25 → RRF fusion → cross-encoder reranking → JSON chunks
```

### Pipeline

1. **Download**: `yt-dlp` fetches auto-generated English subtitles (SRT format) via `--cookies-from-browser chrome`
2. **Parse & chunk**: SRT segments are deduplicated, stripped of HTML tags, and grouped into ~400-word chunks at segment boundaries. Each chunk is prepended with metadata: `search_document: [Video: {title} | Time: HH:MM:SS]`
3. **Index**: Chunks are embedded with `nomic-ai/nomic-embed-text-v1.5` (8192-token context) into a FAISS flat inner-product index. A BM25 sparse index is built in parallel
4. **Retrieve**: Queries hit both dense (FAISS cosine) and sparse (BM25) indexes, fused via Reciprocal Rank Fusion (`K=60`)
5. **Rerank**: Top-20 RRF candidates are rescored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, returning the top-N results

### Models

| Component | Model | Size | Context |
|-----------|-------|------|---------|
| Embedding | `nomic-ai/nomic-embed-text-v1.5` | ~522 MB | 8192 tokens |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | ~6 MB | 512 tokens |

Models are downloaded from HuggingFace Hub on first run and cached at `~/.cache/huggingface/hub/`. The embedding model requires `trust_remote_code=True` (custom NomicBert architecture).

### Rate limiting

A centrifugal governor circuit breaker tracks consecutive yt-dlp failures. After 3 consecutive 429 errors, all downloads pause for 120 seconds. Individual videos retry with exponential backoff (5 attempts, 4–60s).

### Sessions

Sessions are **content-addressed** by video IDs — `SHA256(sorted_video_ids)[:8]`. Same URLs always reuse the same index, even across restarts. Sessions auto-expire after 24 hours. Cache lives at `~/.cache/yt-search/`.

Each session directory contains:
- `meta.json` — creation time, video IDs, titles, chunk count
- `index.faiss` — FAISS flat inner-product index
- `data.pkl` — BM25 index + chunk list (pickled)

## Installation

```bash
pip install -e .
yt-search-install-skill   # installs Claude Code skill to ~/.claude/skills/yt-search/
```

Or symlink the skill for development:
```bash
ln -sf "$(pwd)/skill/yt-search" ~/.claude/skills/yt-search
```

Requires Python 3.11+, `yt-dlp` (installed as a dependency), and internet access for subtitle download and first-run model download.

### Dependencies

| Package | Purpose |
|---------|---------|
| `click` | CLI framework |
| `yt-dlp` | YouTube subtitle download |
| `sentence-transformers` | Embedding and cross-encoder models |
| `faiss-cpu` | Dense vector indexing |
| `rank-bm25` | Sparse BM25 indexing |
| `numpy` | Numerical operations |
| `tenacity` | Retry logic with exponential backoff |

### Known issues

- **macOS/ARM segfault**: `faiss-cpu` and the nomic custom model code conflict if faiss is imported before the embedding model loads. The CLI handles this by preloading the embedding model before any faiss import. If you use the library programmatically, call `from yt_search.models import embed; embed()` before importing faiss.
- **First-run download**: The embedding model (~522 MB) downloads from HuggingFace on first use. The `huggingface_hub` library can hang on unstable connections due to missing read timeouts — the tool configures a 300s read timeout via `set_client_factory` to mitigate this.

## Usage

```bash
# Build an index from one or more videos
yt-search download "https://youtube.com/watch?v=..."
# → a3f2c1b4

# Query the index (returns JSON chunks with timestamps and scores)
yt-search query a3f2c1b4 "what does he say about attention mechanisms?"

# Get more candidates
yt-search query a3f2c1b4 "transformer training tricks" --top-n 10

# List active sessions
yt-search list

# Clean up
yt-search clear a3f2c1b4
yt-search clear --all
```

### Output format

```json
[
  {
    "raw": "the attention mechanism computes a weighted sum...",
    "video": "Andrej Karpathy - Let's build GPT",
    "video_id": "kCc8FmEb1nY",
    "timestamp": "00:23:41",
    "score": 8.34
  }
]
```

### Multi-query strategy

- Start broad, then narrow based on what comes back
- Rephrase if results are weak — retrieval uses dense + sparse fusion (RRF)
- Use `--top-n 10` when coverage matters more than precision
- Multiple URLs in one `download` call are all indexed together in one session

## Claude Code skill

Once installed, Claude Code picks up the skill automatically. When you ask Claude to research YouTube content it will:

1. Call `yt-search download` to build the index
2. Call `yt-search query` iteratively with different phrasings
3. Synthesize an answer from the returned chunks with source citations

## Project structure

```
yt_search/              Python package
  cli.py                Click CLI — download, query, list, clear commands
  ingest.py             yt-dlp download, SRT parsing, chunking, FAISS+BM25 index building
  retrieval.py          Dense+sparse fusion (RRF), cross-encoder reranking
  models.py             Lazy-loaded embedding & reranking models with HF cache awareness
  session.py            Content-addressed session management with 24h TTL
  governor.py           Centrifugal governor circuit breaker for rate limiting
  install_skill.py      Copies skill to ~/.claude/skills/
skill/yt-search/        Agent skill definition
  SKILL.md              Skill metadata and usage guide for Claude Code
  references/           Technical reference docs
docs/                   Research notes and architecture decisions
pyproject.toml          Package definition and dependencies
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
