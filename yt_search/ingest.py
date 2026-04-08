import pickle, re, subprocess, tempfile
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from yt_search.governor import CentrifugalGovernor
from yt_search.models import embed

# One governor per process — tracks rate-limit pressure across all video downloads.
_governor = CentrifugalGovernor(max_swing_height=3, spindown_seconds=120)


class RateLimitError(Exception):
    pass

CHUNK_WORDS = 200
OVERLAP_WORDS = 40

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _strip_tags(text):
    return re.sub(r"<[^>]+>", "", text).strip()


def tokenize(text):
    """Normalize text for BM25: lowercase, strip punctuation, split."""
    return _PUNCT_RE.sub("", text.lower()).split()


def _parse_srt(content):
    segs, seen = [], set()
    for block in re.split(r"\n\n+", content.strip()):
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if len(lines) < 3:
            continue
        m = re.match(r"(\d{2}:\d{2}:\d{2})", lines[1])
        if not m:
            continue
        text = _strip_tags(" ".join(lines[2:]))
        if text and text not in seen:
            seen.add(text)
            segs.append((m.group(1), text))
    return segs


def _build_chunks(segs, title, vid_id):
    chunks, buf, ts0 = [], [], None
    for ts, text in segs:
        if ts0 is None:
            ts0 = ts
        buf.append((ts, text))
        words = " ".join(t for _, t in buf).split()
        if len(words) >= CHUNK_WORDS:
            raw = " ".join(t for _, t in buf)
            chunks.append({
                "text": f"search_document: [Video: {title} | Time: {ts0}]\n{raw}",
                "raw": raw,
                "video": title,
                "video_id": vid_id,
                "timestamp": ts0,
            })
            # Keep trailing segments whose words fit within OVERLAP_WORDS
            overlap_buf, overlap_wc = [], 0
            for seg in reversed(buf):
                seg_wc = len(seg[1].split())
                if overlap_wc + seg_wc > OVERLAP_WORDS:
                    break
                overlap_buf.append(seg)
                overlap_wc += seg_wc
            overlap_buf.reverse()
            buf = overlap_buf
            ts0 = buf[0][0] if buf else None
    if buf:
        raw = " ".join(t for _, t in buf)
        chunks.append({
            "text": f"search_document: [Video: {title} | Time: {ts0}]\n{raw}",
            "raw": raw,
            "video": title,
            "video_id": vid_id,
            "timestamp": ts0,
        })
    return chunks


def _download_subtitles(url: str, tmp: str) -> None:
    """
    Download subtitles for one URL.

    Tenacity handles per-video exponential backoff on 429s.
    The module-level governor cuts all downloads if consecutive videos
    keep failing — the centrifugal balls have swung too high.
    """
    _governor.wait_if_choked()

    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _attempt() -> None:
        result = subprocess.run(
            [
                "yt-dlp", "--write-auto-sub", "--sub-lang", "en",
                "--skip-download", "--sub-format", "srt/best",
                "--cookies-from-browser", "chrome",
                "--ignore-no-formats-error",
                "-o", f"{tmp}/%(id)s|||%(title)s", url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            if "429" in result.stderr or "Too Many Requests" in result.stderr:
                raise RateLimitError(f"YouTube rate-limited ({url})")
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)

    try:
        _attempt()
        _governor.steady_state()
    except RateLimitError:
        _governor.overspeed_surge()
        raise


def download(urls):
    chunks = []
    with tempfile.TemporaryDirectory() as tmp:
        seen_srts: set[Path] = set()
        for url in urls:
            _download_subtitles(url, tmp)
            for srt_f in Path(tmp).glob("*.srt"):
                if srt_f in seen_srts:
                    continue
                seen_srts.add(srt_f)
                base = re.sub(r"\.[a-z]{2}(-[A-Z]{2})?$", "", srt_f.stem)
                parts = base.split("|||", 1)
                vid_id, title = parts[0], (parts[1] if len(parts) > 1 else parts[0])
                chunks.extend(_build_chunks(_parse_srt(srt_f.read_text()), title, vid_id))
    return chunks


def build_index(chunks, session_path):
    import faiss

    texts = [c["text"] for c in chunks]
    model = embed()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    bm25 = BM25Okapi([tokenize(c["raw"]) for c in chunks])

    faiss.write_index(index, str(session_path / "index.faiss"))
    with open(session_path / "data.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "bm25": bm25}, f)
