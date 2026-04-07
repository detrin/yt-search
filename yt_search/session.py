import hashlib, json, shutil, time
from pathlib import Path

CACHE = Path.home() / ".cache" / "yt-search"
TTL = 86400


def sid(video_ids):
    key = ",".join(sorted(video_ids)).encode()
    return hashlib.sha256(key).hexdigest()[:8]


def path(session_id):
    return CACHE / session_id


def load(session_id):
    p = path(session_id) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def save(session_id, data):
    p = path(session_id)
    p.mkdir(parents=True, exist_ok=True)
    (p / "meta.json").write_text(json.dumps(data))


def expired(meta):
    return time.time() - meta["created"] > TTL


def purge_expired():
    if not CACHE.exists():
        return
    for d in CACHE.iterdir():
        if d.is_dir():
            m = load(d.name)
            if m and expired(m):
                shutil.rmtree(d)


def list_all():
    purge_expired()
    if not CACHE.exists():
        return []
    return [(d.name, load(d.name)) for d in CACHE.iterdir() if d.is_dir() and load(d.name)]
