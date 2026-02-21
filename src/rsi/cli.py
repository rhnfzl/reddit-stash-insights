"""CLI entry point for reddit-stash-insights."""
from __future__ import annotations

import sys
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


# ---------------------------------------------------------------------------
# Chat helpers
# ---------------------------------------------------------------------------

def _build_chat_engine(
    *,
    provider: str,
    model: str,
    db_path: Path,
    mode: str,
    max_history: int,
    context_docs: int,
) -> "DirectEngine":  # noqa: F821 — forward ref resolved at runtime
    """Construct a :class:`DirectEngine` from CLI / config parameters."""
    from rsi.chat.engine import DirectEngine
    from rsi.chat.providers.availability import find_available_provider
    from rsi.chat.providers.base import create_provider
    from rsi.indexer.search import SearchEngine, SearchMode

    search_engine = SearchEngine(db_path=db_path)
    resolved_provider, resolved_model, note = find_available_provider(provider, model)
    if resolved_provider is None:
        raise RuntimeError(f"No LLM provider available: {note}")
    if note:
        typer.echo(note)
    llm = create_provider(provider=resolved_provider, model=resolved_model)
    return DirectEngine(
        search_engine=search_engine,
        llm=llm,
        search_mode=SearchMode(mode),
        max_history_turns=max_history,
    )


def _print_sources(sources: list[dict]) -> None:
    """Pretty-print source documents."""
    if not sources:
        typer.echo("  No sources available.")
        return
    typer.echo()
    for i, src in enumerate(sources, 1):
        sub = src.get("subreddit", "?")
        fpath = src.get("file_path", "?")
        preview = src.get("text", "")[:100].replace("\n", " ")
        typer.echo(f"  [{i}] r/{sub} -- {fpath}")
        typer.echo(f"      {preview}")


_REPL_HELP = """\
Commands:
  /help     Show this message
  /clear    Clear conversation history
  /sources  Show sources from the last answer
  /quit     Exit chat (or /exit)
"""


def _run_repl(engine: "DirectEngine", *, limit: int, stream: bool) -> None:  # noqa: F821
    """Interactive read-eval-print loop."""
    typer.echo("rsi chat  --  type /help for commands, /quit to exit\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break

        if not line:
            continue

        # Slash commands
        if line.startswith("/"):
            cmd = line.lower()
            if cmd in ("/quit", "/exit"):
                break
            if cmd == "/help":
                typer.echo(_REPL_HELP)
                continue
            if cmd == "/clear":
                engine.clear_history()
                typer.echo("History cleared.\n")
                continue
            if cmd == "/sources":
                sources = engine.history.last_sources()
                if not sources:
                    typer.echo("No sources yet.\n")
                else:
                    _print_sources(sources)
                    typer.echo()
                continue
            typer.echo(f"Unknown command: {line}  (type /help)\n")
            continue

        # Regular question
        if stream:
            _stream_answer(engine, line, limit=limit)
        else:
            resp = engine.chat(line, limit=limit)
            typer.echo(f"\n{resp.answer}")
            _print_sources(resp.sources)
            typer.echo()


def _stream_answer(engine: "DirectEngine", query: str, *, limit: int) -> None:  # noqa: F821
    """Stream tokens to stdout, then print sources."""
    token_iter, sources = engine.chat_stream(query, limit=limit)
    typer.echo()
    for token in token_iter:
        sys.stdout.write(token)
        sys.stdout.flush()
    typer.echo()
    _print_sources(sources)
    typer.echo()


@app.command()
def chat(
    question: Optional[str] = typer.Argument(None, help="Question to ask (omit for interactive REPL)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider: ollama, llama-cpp, openai"),
    model: Optional[str] = typer.Option(None, "--model", help="Model name or path"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to vector database"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Number of context documents"),
    mode: Optional[str] = typer.Option(None, "--mode", "-m", help="Search mode: hybrid, semantic, keyword"),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming (print complete response)"),
) -> None:
    """Ask questions about your Reddit archive using RAG chat."""
    from rsi.config import Settings

    settings = Settings.load()

    # Resolve options: CLI flag > config > default
    _provider = provider or settings.llm_provider
    _model = model or settings.llm_model
    _db_path = db_path or settings.db_path
    _mode = mode or settings.chat_search_mode
    _limit = limit if limit is not None else settings.chat_context_docs

    engine = _build_chat_engine(
        provider=_provider,
        model=_model,
        db_path=_db_path,
        mode=_mode,
        max_history=settings.chat_max_history,
        context_docs=settings.chat_context_docs,
    )

    stream = not no_stream

    if question:
        # Single-turn mode
        if stream:
            _stream_answer(engine, question, limit=_limit)
        else:
            resp = engine.chat(question, limit=_limit)
            typer.echo(resp.answer)
            _print_sources(resp.sources)
    else:
        # Interactive REPL
        _run_repl(engine, limit=_limit, stream=stream)


@app.command()
def mcp(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or sse"),
) -> None:
    """Start the MCP server for Claude Code integration."""
    import asyncio

    from rsi.mcp_server import run_stdio

    if transport == "stdio":
        asyncio.run(run_stdio())
    else:
        typer.echo(f"Transport '{transport}' not yet supported. Use 'stdio'.")
        raise typer.Exit(1)
