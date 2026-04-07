_embed = None
_rerank = None


def embed():
    global _embed
    if _embed is None:
        from sentence_transformers import SentenceTransformer
        _embed = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return _embed


def rerank():
    global _rerank
    if _rerank is None:
        from sentence_transformers import CrossEncoder
        _rerank = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _rerank
