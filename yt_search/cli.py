import json, shutil, sys, time

import click

import yt_search.session as sess
from yt_search.ingest import download, build_index
from yt_search.retrieval import load, search


def _preload_models():
    """Load embedding model before faiss to avoid macOS/ARM segfault.

    faiss-cpu and the nomic custom model code conflict when faiss is
    imported first. Calling this before any faiss usage prevents the crash.
    """
    from yt_search.models import embed
    embed()


@click.group()
def main():
    pass


@main.command()
@click.argument("urls", nargs=-1, required=True)
def download_cmd(urls):
    sess.purge_expired()
    _preload_models()
    chunks = download(list(urls))
    if not chunks:
        click.echo("No subtitles found.", err=True)
        sys.exit(1)

    video_ids = list({c["video_id"] for c in chunks})
    session_id = sess.sid(video_ids)
    sp = sess.path(session_id)

    if sp.exists() and not sess.expired(sess.load(session_id) or {"created": 0}):
        click.echo(session_id)
        return

    sp.mkdir(parents=True, exist_ok=True)
    build_index(chunks, sp)
    sess.save(session_id, {
        "created": time.time(),
        "videos": video_ids,
        "titles": list({c["video"] for c in chunks}),
        "chunks": len(chunks),
    })
    click.echo(session_id)


@main.command()
@click.argument("session_id")
@click.argument("query")
@click.option("--top-n", default=5, show_default=True)
def query(session_id, query, top_n):
    _preload_models()
    meta = sess.load(session_id)
    if not meta:
        click.echo(f"Session {session_id} not found.", err=True)
        sys.exit(1)
    if sess.expired(meta):
        click.echo(f"Session {session_id} expired. Re-run download.", err=True)
        sys.exit(1)

    index, bm25, chunks = load(sess.path(session_id))
    results = search(query, index, bm25, chunks, top_n=top_n)
    click.echo(json.dumps(results, indent=2))


@main.command("list")
def list_cmd():
    sessions = sess.list_all()
    if not sessions:
        click.echo("No active sessions.")
        return
    for sid, meta in sessions:
        age_h = (time.time() - meta["created"]) / 3600
        titles = ", ".join(meta.get("titles", []))
        click.echo(f"{sid}  [{age_h:.1f}h old]  {meta.get('chunks', '?')} chunks  {titles}")


@main.command()
@click.argument("session_id", required=False)
@click.option("--all", "all_", is_flag=True, help="Clear all sessions.")
def clear(session_id, all_):
    if all_:
        shutil.rmtree(sess.CACHE, ignore_errors=True)
        click.echo("Cleared all sessions.")
    elif session_id:
        shutil.rmtree(sess.path(session_id), ignore_errors=True)
        click.echo(f"Cleared {session_id}.")
    else:
        click.echo("Specify a session-id or --all.", err=True)
        sys.exit(1)


main.add_command(download_cmd, name="download")
