"""CLI entry point for reddit-stash-insights."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

if TYPE_CHECKING:
    from rsi.chat.engine import DirectEngine

from rsi import __version__

app = typer.Typer(
    name="rsi",
    help="Semantic search and AI chat for reddit-stash archives.",
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
    """Search and chat with your Reddit archive."""
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


# Exit codes shared by index, search and health (2 is typer's usage error).
EXIT_UNHEALTHY = 1
EXIT_DATA_GATE = 3
EXIT_MISMATCH = 4


@app.command()
def index(
    reddit_dir: Path = typer.Argument(..., help="Path to reddit-stash output directory", exists=True),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path for vector database (default: RSI_DB_PATH)"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket containing reddit-stash data"),
    s3_prefix: str = typer.Option("reddit/", "--s3-prefix", help="S3 key prefix for reddit-stash files"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop the existing index first (needed after an embedder change)"),
    max_error_pct: Optional[float] = typer.Option(
        None, "--max-error-pct",
        help="Exit 3 without writing metadata when nothing parses or parse errors exceed this percentage",
    ),
):
    """Build or update the vector search index from reddit-stash content."""
    from rsi.config import Settings
    from rsi.core.scanner import scan_directory
    from rsi.indexer.index_meta import IndexConfigMismatch, remove_meta, write_meta
    from rsi.indexer.search import SearchEngine

    settings = Settings.load()

    # If S3 bucket specified, fetch files first
    if s3_bucket:
        from rsi.core.s3_fetch import fetch_from_s3
        typer.echo(f"Fetching from s3://{s3_bucket}/{s3_prefix}...")
        reddit_dir = fetch_from_s3(bucket=s3_bucket, prefix=s3_prefix)
        typer.echo(f"Files cached to {reddit_dir}")

    db_path = db_path or settings.db_path

    typer.echo(f"Scanning {reddit_dir}...")
    scan_result = scan_directory(reddit_dir)
    n_items = len(scan_result.posts) + len(scan_result.comments)
    n_errors = len(scan_result.errors)
    typer.echo(f"Found {len(scan_result.posts)} posts, {len(scan_result.comments)} comments, {n_errors} parse errors")

    if max_error_pct is not None and (n_items == 0 or n_errors * 100 > max_error_pct * (n_items + n_errors)):
        typer.echo(f"Refusing to index: {n_items} items, {n_errors} parse errors (limit {max_error_pct}%)", err=True)
        raise typer.Exit(EXIT_DATA_GATE)

    if not n_items:
        typer.echo("Nothing to index.")
        raise typer.Exit()

    try:
        # allow_incomplete: re-running over a crashed build resumes it (rows are upserted).
        engine = SearchEngine(db_path=db_path, settings=settings, verify=not rebuild, allow_incomplete=True, writing=True)
    except IndexConfigMismatch as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(EXIT_MISMATCH)
    from itertools import chain

    provider, model = engine.embedder_identity()
    if provider == "local":
        typer.echo("Loading embedding model (first run downloads ~2GB)...")
    try:
        input_token_capacity = engine.validate_index_settings()
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2)
    if rebuild:
        remove_meta(db_path)
        engine.store.drop()

    typer.echo(f"Embedding passages from {n_items} items with {provider}/{model}...")
    # Readers must refuse partial rows until embedding and the full-text index both finish.
    write_meta(db_path, provider=provider, model=model, dim=0, rows=0, parse_errors=n_errors,
               complete=False, chunk_tokens=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap,
               input_token_capacity=input_token_capacity)
    dim = engine.index_items(chain(scan_result.posts, scan_result.comments))
    write_meta(db_path, provider=provider, model=model, dim=dim, rows=engine.store.count(), parse_errors=n_errors,
               chunk_tokens=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap,
               input_token_capacity=input_token_capacity)
    typer.echo(f"Indexed {n_items} items as {engine.store.count()} passages to {db_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to vector database (default: RSI_DB_PATH)"),
    subreddit: Optional[str] = typer.Option(None, "--subreddit", "-s", help="Filter by subreddit"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, semantic, keyword"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
):
    """Search your indexed reddit-stash content."""
    from rsi.config import Settings
    from rsi.indexer.index_meta import IndexConfigMismatch, IndexIncomplete
    from rsi.indexer.search import SearchEngine, SearchMode

    settings = Settings.load()
    db_path = db_path or settings.db_path

    if not db_path.exists():
        typer.echo("No index found. Run 'rsi index' first. 0 results found.")
        raise typer.Exit(EXIT_UNHEALTHY)

    search_mode = SearchMode(mode)
    try:
        engine = SearchEngine(db_path=db_path, settings=settings)
    except IndexConfigMismatch as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(EXIT_MISMATCH)
    except IndexIncomplete as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(EXIT_UNHEALTHY)
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


@app.command()
def health(
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to vector database (default: RSI_DB_PATH)"),
    mode: str = typer.Option("keyword", "--mode", "-m", help="keyword (no embedding call), hybrid or semantic"),
) -> None:
    """Self-test an index: exit 0 healthy, 1 unhealthy, 4 built with another embedder."""
    import re

    from rsi.config import Settings
    from rsi.indexer.index_meta import IndexConfigMismatch, IndexIncomplete
    from rsi.indexer.search import SearchEngine, SearchMode

    settings = Settings.load()
    db_path = db_path or settings.db_path
    search_mode = SearchMode(mode)

    def unhealthy(reason: str) -> None:
        typer.echo(f"unhealthy: {reason}", err=True)
        raise typer.Exit(EXIT_UNHEALTHY)

    if not db_path.exists():
        unhealthy(f"no index at {db_path}")
    try:
        engine = SearchEngine(db_path=db_path, settings=settings)
    except IndexConfigMismatch as e:
        typer.echo(f"unhealthy: {e}", err=True)
        raise typer.Exit(EXIT_MISMATCH)
    except IndexIncomplete as e:
        unhealthy(str(e))
    rows = engine.store.count()
    if rows == 0:
        unhealthy(f"index at {db_path} has no rows")
    # Probe with words taken from stored rows, longest first: the FTS tokenizer drops
    # stop words (mostly short) and tokens over 40 chars, so a working index has a hit.
    words = {w for t in engine.store.sample_texts(20) for w in re.findall(r"[^\W_]+", t) if len(w) <= 40}
    # Try many words: a short sample can be mostly stop words (about, between, through, ...)
    probes = sorted(words, key=lambda w: (-len(w), w))[:25]
    if not probes:
        unhealthy("could not pick a probe word from the index")
    for word in probes:
        try:
            hits = engine.search(word, limit=1, mode=search_mode)
        except Exception as e:  # any search failure means the server cannot serve
            unhealthy(f"{mode} search failed: {e}")
        if hits:
            break
    else:
        unhealthy(f"{mode} search for {probes} returned nothing")
    provider, model = engine.embedder_identity()
    typer.echo(f"healthy: {rows} rows, {provider}/{model}, {mode} search ok")


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
) -> DirectEngine:
    """Construct a :class:`DirectEngine` from CLI / config parameters."""
    from rsi.chat.engine import DirectEngine
    from rsi.chat.providers.availability import find_available_provider, provider_destination
    from rsi.chat.providers.base import create_provider
    from rsi.config import Settings
    from rsi.indexer.search import SearchEngine, SearchMode

    settings = Settings.load()
    resolved_provider, resolved_model, note = find_available_provider(
        provider, model, allow_cloud_fallback=settings.allow_cloud_fallback,
    )
    if resolved_provider is None:
        raise RuntimeError(f"No LLM provider available: {note}")
    assert resolved_model is not None  # guaranteed when provider is not None
    if note:
        typer.echo(note)
    typer.echo(f"Chat context destination: {provider_destination(resolved_provider)} (model {resolved_model})")
    search_engine = SearchEngine(db_path=db_path, settings=settings)
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


def _run_repl(engine: DirectEngine, *, limit: int, stream: bool) -> None:
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


def _stream_answer(engine: DirectEngine, query: str, *, limit: int) -> None:
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
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or http"),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP bind address"),
    port: int = typer.Option(8765, "--port", help="HTTP port"),
) -> None:
    """Start the MCP server (stdio for local clients, http for remote clients)."""
    if transport == "stdio":
        import asyncio

        from rsi.mcp_server import run_stdio

        asyncio.run(run_stdio())
    elif transport == "http":
        from rsi.indexer.index_meta import IndexConfigMismatch
        from rsi.mcp_server import run_http

        try:
            run_http(host=host, port=port)
        except IndexConfigMismatch as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(EXIT_MISMATCH)
        except RuntimeError as e:
            typer.echo(str(e), err=True)
            raise typer.Exit(1)
    else:
        typer.echo(f"Transport '{transport}' not supported. Use 'stdio' or 'http'.")
        raise typer.Exit(1)


@app.command("eval")
def evaluate_search(
    dataset: Optional[Path] = typer.Option(None, "--dataset", help="Questions JSONL; snapshot.json must be alongside"),
    qrels: Optional[Path] = typer.Option(None, "--qrels", help="Canonical source-thread relevance labels JSONL"),
    corpus: Optional[Path] = typer.Option(None, "--corpus", help="Frozen Markdown corpus to verify against snapshot.json"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Explicit evaluation index; never defaults to live index"),
    mode: str = typer.Option("keyword", "--mode", help="keyword, semantic or hybrid"),
    split: str = typer.Option("dev", "--split", help="dev, test or all"),
    output: Path = typer.Option(Path("evaluation.json"), "--output", help="JSON report destination"),
    fixture: bool = typer.Option(False, "--fixture", help="Build a temporary FTS index of the packaged synthetic corpus"),
    allow_remote: bool = typer.Option(False, "--allow-remote", help="Allow the configured hosted embedding destination"),
) -> None:
    """Measure search against frozen labels without calling a chat model."""
    import hashlib
    import json
    import os
    import tempfile
    from urllib.parse import urlsplit, urlunsplit

    from rsi.config import Settings
    from rsi.evaluation import evaluate, export_fixture, index_fingerprint, load_dataset
    from rsi.indexer.index_meta import read_meta

    if mode not in ("keyword", "semantic", "hybrid") or split not in ("dev", "test", "all"):
        raise typer.BadParameter("mode must be keyword/semantic/hybrid and split must be dev/test/all")
    if fixture and (mode != "keyword" or any(value is not None for value in (dataset, qrels, corpus, db_path))):
        raise typer.BadParameter("--fixture requires keyword mode and owns its temporary corpus and index")
    if not fixture and any(value is None for value in (dataset, qrels, corpus, db_path)):
        raise typer.BadParameter("provide --dataset, --qrels, --corpus and --db-path, or use --fixture")
    settings = Settings.load()
    if mode != "keyword" and settings.embedding_provider != "local":
        destination = settings.embedding_base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        parts = urlsplit(destination)
        destination = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
        if not allow_remote:
            raise typer.BadParameter(f"hosted embeddings require --allow-remote; destination: {destination}")
        typer.echo(f"Embedding destination: {destination}. Evaluation query text will be sent there.", err=True)
    try:
        with tempfile.TemporaryDirectory(prefix="rsi-eval-") as tmp:
            if fixture:
                from rsi.core.scanner import scan_directory
                from rsi.indexer.vector_store import VectorStore

                fixture_root = Path(tmp) / "fixture"
                export_fixture(fixture_root)
                dataset, qrels, corpus = (fixture_root / "questions.jsonl", fixture_root / "qrels.jsonl",
                                           fixture_root / "corpus")
                db_path = Path(tmp) / "index"
                queries, labels, snapshot = load_dataset(dataset, qrels, corpus)
                scan_result = scan_directory(corpus)
                store = VectorStore(db_path)
                records: list[dict] = []
                for kind, items in (("post", scan_result.posts), ("comment", scan_result.comments)):
                    records.extend({"id": item.id, "content_type": kind, "text": item.search_text(),
                                    "subreddit": item.subreddit, "permalink": item.permalink,
                                    "file_path": str(item.file_path)} for item in items)
                store.add_records(records)
                store.create_fts_index()
                def search(query, limit, mode, subreddit):
                    return store.fts_search(query, limit, subreddit)
                index_identity = {"schema": "FTS-only synthetic fixture", "rows": store.count(),
                                  **index_fingerprint(store, snapshot)}
            else:
                from rsi.indexer.search import SearchEngine, SearchMode

                assert dataset is not None and qrels is not None and corpus is not None and db_path is not None
                queries, labels, snapshot = load_dataset(dataset, qrels, corpus)
                if not db_path.exists():
                    raise ValueError(f"evaluation index does not exist: {db_path}")
                metadata = read_meta(db_path)
                if metadata is not None and not isinstance(metadata, dict):
                    raise ValueError("evaluation index metadata must be a JSON object")
                engine = SearchEngine(db_path=db_path, settings=settings)
                def search(query, limit, mode, subreddit):
                    return engine.search(query, limit=limit, mode=SearchMode(mode), subreddit=subreddit)
                index_identity = {**(metadata or {"schema": "legacy", "rows": engine.store.count()}),
                                  **index_fingerprint(engine.store, snapshot)}
            identity = {"embedding": "none (keyword)" if mode == "keyword" else {
                "provider": settings.embedding_provider, "model": settings.embedding_model,
            }, "index": index_identity, "config": {"mode": mode, "split": split, "cutoff": 10,
                  "candidate_depths": [50, 100, 200], "thread_order": "first occurrence",
                  "allow_remote": allow_remote, "chunk_tokens": settings.chunk_tokens,
                  "chunk_overlap": settings.chunk_overlap,
                  "index_chunk_size": settings.index_chunk_size, "embedding_use_fp16": settings.embedding_use_fp16},
                "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "qrels_sha256": hashlib.sha256(qrels.read_bytes()).hexdigest()}
            report = evaluate(queries, labels, snapshot, search, mode=mode, split=split, identity=identity)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        typer.echo(f"Evaluation failed: {error}", err=True)
        raise typer.Exit(1)
    typer.echo(f"{report['answerable_count']} answerable, {report['absent_count']} absent, "
               f"{report['error_count']} errors. Report: {output}")
    if report["error_count"]:
        raise typer.Exit(1)


@app.command()
def ui(
    port: int = typer.Option(8501, "--port", min=1, max=65535, help="Local web interface port"),
) -> None:
    """Open the installed web interface on this computer."""
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        typer.echo("Install UI support with pip install 'reddit-stash-insights[ui]'", err=True)
        raise typer.Exit(1)
    script = Path(__file__).parent / "ui" / "chat.py"
    try:
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(script),
            "--server.address", "127.0.0.1", "--server.port", str(port),
        ])
    except KeyboardInterrupt:
        raise typer.Exit(130)
    raise typer.Exit(result.returncode)
