"""CLI entry point for reddit-stash-insights."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from rsi import __version__

app = typer.Typer(
    name="rsi",
    help="Semantic search, knowledge graph, and AI chat for reddit-stash archives.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"reddit-stash-insights {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True),
):
    """reddit-stash-insights: search, graph, analytics, and chat for your Reddit archive."""
    pass


@app.command()
def scan(
    reddit_dir: Path = typer.Argument(..., help="Path to reddit-stash output directory", exists=True),
):
    """Scan a reddit-stash output directory and report contents."""
    from rsi.core.scanner import scan_directory

    result = scan_directory(reddit_dir)
    typer.echo(f"Found {len(result.posts)} posts and {len(result.comments)} comments")
    if result.errors:
        typer.echo(f"  ({len(result.errors)} parse errors)", err=True)


@app.command()
def index(
    reddit_dir: Path = typer.Argument(..., help="Path to reddit-stash output directory", exists=True),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path for vector database"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket containing reddit-stash data"),
    s3_prefix: str = typer.Option("reddit/", "--s3-prefix", help="S3 key prefix for reddit-stash files"),
):
    """Build or update the vector search index from reddit-stash content."""
    from rsi.config import DEFAULT_DB_PATH
    from rsi.core.scanner import scan_directory
    from rsi.indexer.search import SearchEngine

    # If S3 bucket specified, fetch files first
    if s3_bucket:
        from rsi.core.s3_fetch import fetch_from_s3
        typer.echo(f"Fetching from s3://{s3_bucket}/{s3_prefix}...")
        reddit_dir = fetch_from_s3(bucket=s3_bucket, prefix=s3_prefix)
        typer.echo(f"Files cached to {reddit_dir}")

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    typer.echo(f"Scanning {reddit_dir}...")
    scan_result = scan_directory(reddit_dir)
    typer.echo(f"Found {len(scan_result.posts)} posts, {len(scan_result.comments)} comments")

    if not scan_result.posts and not scan_result.comments:
        typer.echo("Nothing to index.")
        raise typer.Exit()

    typer.echo("Loading embedding model (first run downloads ~2GB)...")
    engine = SearchEngine(db_path=db_path)

    texts = []
    ids = []
    metadata = []

    for post in scan_result.posts:
        texts.append(post.search_text())
        ids.append(post.id)
        metadata.append({
            "subreddit": post.subreddit,
            "score": post.score,
            "timestamp": str(post.timestamp or ""),
            "file_path": post.file_path,
            "content_type": "post",
        })

    for comment in scan_result.comments:
        texts.append(comment.search_text())
        ids.append(comment.id)
        metadata.append({
            "subreddit": comment.subreddit,
            "score": comment.score,
            "timestamp": str(comment.timestamp or ""),
            "file_path": comment.file_path,
            "content_type": "comment",
        })

    typer.echo(f"Embedding {len(texts)} items...")
    engine.index_texts(texts=texts, ids=ids, metadata=metadata)
    typer.echo(f"Indexed {len(texts)} items to {db_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to vector database"),
    subreddit: Optional[str] = typer.Option(None, "--subreddit", "-s", help="Filter by subreddit"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, semantic, keyword"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
):
    """Search your indexed reddit-stash content."""
    from rsi.config import DEFAULT_DB_PATH
    from rsi.indexer.search import SearchEngine, SearchMode

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not db_path.exists():
        typer.echo("No index found. Run 'rsi index' first. 0 results found.")
        raise typer.Exit()

    search_mode = SearchMode(mode)
    engine = SearchEngine(db_path=db_path)
    results = engine.search(query, limit=limit, mode=search_mode, subreddit=subreddit)

    if not results:
        typer.echo(f"0 results found for: {query}")
        return

    typer.echo(f"{len(results)} results for: {query}\n")
    for i, r in enumerate(results, 1):
        sub = r.get("subreddit", "?")
        score = r.get("score", 0)
        text_preview = r.get("text", "")[:120].replace("\n", " ")
        typer.echo(f"  {i}. [r/{sub}] (score: {score}) {text_preview}")
        typer.echo(f"     File: {r.get('file_path', '?')}")
        typer.echo()
