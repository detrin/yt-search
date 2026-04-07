from pathlib import Path

_embed = None
_rerank = None
_http_configured = False

EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5"
RERANK_REPO = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _resolve_local(repo_id):
    """Find a fully-cached local snapshot for a HF model."""
    slug = "models--" + repo_id.replace("/", "--")
    base = HF_CACHE / slug
    for subdir in ("manual", "snapshots"):
        d = base / subdir
        if not d.exists():
            continue
        candidates = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True) if subdir == "snapshots" else [d]
        for c in candidates:
            if not c.is_dir():
                continue
            has_weights = any(
                f.suffix in (".bin", ".safetensors")
                for f in c.iterdir() if f.is_file()
            )
            if has_weights:
                return str(c)
    return None


def _configure_http():
    global _http_configured
    if not _http_configured:
        import httpx
        from huggingface_hub import set_client_factory
        set_client_factory(
            lambda: httpx.Client(
                timeout=httpx.Timeout(10, read=300),
                follow_redirects=True,
            )
        )
        _http_configured = True


def embed():
    global _embed
    if _embed is None:
        _configure_http()
        from sentence_transformers import SentenceTransformer
        _embed = SentenceTransformer(EMBED_REPO, trust_remote_code=True)
    return _embed


def rerank():
    global _rerank
    if _rerank is None:
        from sentence_transformers import CrossEncoder
        local = _resolve_local(RERANK_REPO)
        if local:
            _rerank = CrossEncoder(local)
        else:
            _rerank = CrossEncoder(RERANK_REPO)
    return _rerank
