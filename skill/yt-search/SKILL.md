---
name: yt-search
description: >
  Use this skill when the user asks you to search YouTube videos, find what someone said
  in a video, research a topic across YouTube content, answer questions from video
  transcripts, or summarize what a YouTube video covers. Also use when the user provides
  a YouTube URL and asks you to extract information from it. You are the reasoning layer —
  this skill gives you a CLI to retrieve relevant transcript chunks; you synthesize the answer.
compatibility: Requires yt-search CLI installed (pip install yt-search). Requires yt-dlp and internet access.
license: MIT
allowed-tools: Bash
---

# yt-search

CLI-based YouTube transcript RAG. Downloads subtitles, builds an ephemeral hybrid index, returns ranked chunks for you to synthesize. You reason; the tool retrieves.

## Commands

```bash
yt-search download <url> [<url> ...]   # build index → prints session-id (content-addressed)
yt-search query <session-id> "<text>"  # retrieve chunks → JSON array
yt-search query <session-id> "<text>" --top-n 10  # more candidates
yt-search list                         # show active sessions with age + chunk count
yt-search clear <session-id>           # remove one session
yt-search clear --all                  # remove all sessions
```

## Session model

Sessions are **content-addressed** by video IDs — same URLs reuse the same index, even across restarts. Sessions **auto-expire after 24 hours**. Cache is at `~/.cache/yt-search/`.

## Workflow

**1. Build the index once:**
```bash
yt-search download "https://youtube.com/watch?v=..."
# → a3f2c1b4
```

**2. Query iteratively — as many times as needed:**
```bash
yt-search query a3f2c1b4 "attention mechanism explanation"
yt-search query a3f2c1b4 "training tricks and hyperparameters"
```

**3. Synthesize the answer yourself** from the returned chunks.

## Output format

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

Cite sources as: `[{video title}, {timestamp}]`

## Multi-query strategy

- Start broad, then narrow based on what comes back
- Rephrase if results are weak — retrieval uses dense + sparse fusion (RRF)
- Use `--top-n 10` when coverage matters more than precision
- Multiple URLs in one `download` call → all indexed together, queryable in one session

## Gotchas

- If `yt-search download` fails with no subtitles, the video may have no auto-generated captions. Try a different video or search for an alternative upload.
- Session IDs are 8-char hex hashes — recheck with `yt-search list` if unsure of the current id.
- First run downloads embedding models (~140MB for nomic-embed, ~6MB for reranker) — expect a longer initial setup.
- Read `references/retrieval-stack.md` for details on the retrieval architecture if you need to explain results or troubleshoot quality issues.
