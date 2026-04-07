# yt-search

Search YouTube video transcripts using a local RAG pipeline exposed as a CLI tool for AI agents.

## What it does

Downloads YouTube subtitles, builds an ephemeral hybrid retrieval index, and returns ranked transcript chunks. Designed to be called by an AI agent (Claude Code) that handles the reasoning and synthesis — the tool handles retrieval only.

## How it works

```
YouTube URL → yt-dlp (SRT) → sentence chunks + metadata → FAISS + BM25
                                                                  ↓
query → nomic-embed-text-v1.5 + BM25 → RRF fusion → cross-encoder reranking → JSON chunks
```

- **Embedding**: `nomic-ai/nomic-embed-text-v1.5` (8192-token context)
- **Retrieval**: FAISS dense + BM25 sparse fused via Reciprocal Rank Fusion
- **Reranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (top-20 → top-5)
- **Sessions**: content-addressed by video IDs, 24h TTL, cached at `~/.cache/yt-search/`

## Installation

```bash
pip install -e .
yt-search-install-skill   # installs Claude Code skill to ~/.claude/skills/yt-search/
```

Or symlink the skill for development:
```bash
ln -sf "$(pwd)/skill/yt-search" ~/.claude/skills/yt-search
```

Requires `yt-dlp` (installed as a dependency) and internet access for subtitle download.

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

## Claude Code skill

Once installed, Claude Code picks up the skill automatically. When you ask Claude to research YouTube content it will:

1. Call `yt-search download` to build the index
2. Call `yt-search query` iteratively with different phrasings
3. Synthesize an answer from the returned chunks with source citations

## Project structure

```
yt_search/       Python package (CLI + RAG pipeline)
skill/           Agent Skills directory (SKILL.md + references/)
docs/            Research notes and architecture decisions
pyproject.toml   Package definition and dependencies
```

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
