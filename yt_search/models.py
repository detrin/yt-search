from pathlib import Path

_embed = None
_rerank = None

EMBED_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_local(repo_id):
    slug = "models--" + repo_id.replace("/", "--")
    base = HF_CACHE / slug
    for subdir in ("manual", "snapshots"):
        d = base / subdir
        if not d.exists():
            continue
        candidates = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if subdir == "snapshots" else [d]
        for c in candidates:
            if c.is_dir() and any(c.iterdir()):
                return str(c)
    return None


def _load(cls, repo_id):
    local = _resolve_local(repo_id)
    if local:
        return cls(local)
    return cls(repo_id)


def embed():
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        _embed = _load(SentenceTransformer, EMBED_REPO)
    return _embed


def rerank():
    global _rerank
    if _rerank is None:
        from sentence_transformers import CrossEncoder
        _rerank = _load(CrossEncoder, RERANK_REPO)
    return _rerank
