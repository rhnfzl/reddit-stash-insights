"""MCP server exposing reddit-stash-insights search and chat as tools.

Two transports: stdio (a local client spawns ``rsi mcp``) and Streamable HTTP
(``rsi mcp --transport http``) for remote clients, with a bearer token.
"""
from __future__ import annotations

import hmac
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

import anyio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

if TYPE_CHECKING:
    from starlette.applications import Starlette

    from rsi.indexer.search import SearchEngine

logger = logging.getLogger(__name__)

MAX_LIMIT = 50
SEARCH_MODES = ("hybrid", "semantic", "keyword")

# ---------------------------------------------------------------------------
# Search engine: opened once (eagerly for http, lazily for stdio)
# ---------------------------------------------------------------------------

_search_engine: SearchEngine | None = None
_search_engine_lock = threading.Lock()


def _get_search_engine() -> SearchEngine:
    global _search_engine  # noqa: PLW0603
    if _search_engine is None:
        # Tool calls run in worker threads; concurrent first calls must open one engine.
        with _search_engine_lock:
            if _search_engine is None:
                from rsi.config import Settings
                from rsi.indexer.search import SearchEngine

                settings = Settings.load()
                _search_engine = SearchEngine(db_path=settings.db_path, settings=settings)
    return _search_engine


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

REDDIT_SEARCH_TOOL = Tool(
    name="reddit_search",
    description=(
        "Search the user's saved Reddit content (posts and comments). "
        "Returns matching documents without LLM generation."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of results to return (1-{MAX_LIMIT})",
                "default": 10,
            },
            "mode": {
                "type": "string",
                "description": "Search mode",
                "enum": list(SEARCH_MODES),
                "default": "hybrid",
            },
        },
        "required": ["query"],
    },
)

REDDIT_CHAT_TOOL = Tool(
    name="reddit_chat",
    description=(
        "Ask a question about the user's saved Reddit content using RAG. "
        "Retrieves relevant documents and generates an answer with an LLM."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Question to ask about the Reddit archive",
            },
            "limit": {
                "type": "integer",
                "description": f"Number of context documents to retrieve (1-{MAX_LIMIT})",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _text(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def _clamp_limit(value: Any, default: int) -> int:
    try:
        return max(1, min(MAX_LIMIT, int(value)))
    except (TypeError, ValueError):
        return default


def _format_search_results(results: list[dict[str, Any]]) -> str:
    """Format search results as a numbered text list."""
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, r in enumerate(results, 1):
        sub = r.get("subreddit", "?")
        score = r.get("score", 0)
        text_preview = r.get("text", "")[:200].replace("\n", " ")
        file_path = r.get("file_path", "?")
        lines.append(f"{i}. [r/{sub}] (score: {score}) {text_preview}")
        lines.append(f"   File: {file_path}")
    return "\n".join(lines)


async def _handle_reddit_search(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle ``reddit_search``: retrieval only, no chat LLM needed."""
    from rsi.indexer.search import SearchMode

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _text("Error: 'query' must be a non-empty string.")
    limit = _clamp_limit(arguments.get("limit", 10), 10)
    mode_str = arguments.get("mode", "hybrid")
    if mode_str not in SEARCH_MODES:
        return _text(f"Error: 'mode' must be one of {', '.join(SEARCH_MODES)}.")

    def run() -> list[dict[str, Any]]:
        return _get_search_engine().search(query, limit=limit, mode=SearchMode(mode_str))

    return _text(_format_search_results(await anyio.to_thread.run_sync(run)))


def _chat_sync(query: str, limit: int) -> str:
    """Answer one question with a fresh engine, so no history leaks between calls or clients."""
    from rsi.chat.engine import DirectEngine
    from rsi.chat.providers.availability import find_available_provider
    from rsi.chat.providers.base import create_provider
    from rsi.config import Settings
    from rsi.indexer.search import SearchMode

    settings = Settings.load()
    provider, model, note = find_available_provider(settings.llm_provider, settings.llm_model)
    if provider is None or model is None:
        return f"reddit_chat is unavailable: {note} Use reddit_search instead."
    engine = DirectEngine(
        search_engine=_get_search_engine(),
        llm=create_provider(provider=provider, model=model),
        search_mode=SearchMode(settings.chat_search_mode),
        max_history_turns=settings.chat_max_history,
    )
    resp = engine.chat(query, limit=limit)

    parts = [resp.answer]
    if resp.sources:
        parts.append("\nSources:")
        for i, src in enumerate(resp.sources, 1):
            parts.append(f"  [{i}] r/{src.get('subreddit', '?')} -- {src.get('file_path', '?')}")
    return "\n".join(parts)


async def _handle_reddit_chat(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle ``reddit_chat``."""
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return _text("Error: 'query' must be a non-empty string.")
    limit = _clamp_limit(arguments.get("limit", 5), 5)
    return _text(await anyio.to_thread.run_sync(_chat_sync, query, limit))


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "reddit_search": _handle_reddit_search,
    "reddit_chat": _handle_reddit_chat,
}


def create_server() -> Server:
    """Build and return a configured MCP :class:`Server`."""
    server = Server("reddit-stash-insights")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [REDDIT_SEARCH_TOOL, REDDIT_CHAT_TOOL]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = _TOOL_HANDLERS.get(name)
        if handler is None:
            return _text(f"Unknown tool: {name}")
        return await handler(arguments)

    return server


# ---------------------------------------------------------------------------
# Stdio runner
# ---------------------------------------------------------------------------


async def run_stdio() -> None:
    """Run the MCP server over stdio transport."""
    server = create_server()
    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options)


# ---------------------------------------------------------------------------
# Streamable HTTP runner
# ---------------------------------------------------------------------------

HEALTH_PATH = "/healthz"


class BearerAuthMiddleware:
    """Require ``Authorization: Bearer <token>`` on every HTTP path except the health probe."""

    def __init__(self, app: Any, token: str):
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and scope["path"] != HEALTH_PATH:
            auth = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(auth, self._expected):
                from starlette.responses import PlainTextResponse

                resp = PlainTextResponse("unauthorized", status_code=401, headers={"WWW-Authenticate": "Bearer"})
                await resp(scope, receive, send)
                return
        await self.app(scope, receive, send)


def _csv_env(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


def create_http_app(*, token: str, allowed_hosts: list[str]) -> Starlette:
    """Starlette app: ``/mcp`` (Streamable HTTP, SSE responses) behind bearer auth, open ``/healthz``.

    ``allowed_hosts`` empty disables Host/Origin checks (the bearer token still applies).
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(allowed_hosts),
        allowed_hosts=allowed_hosts,
        allowed_origins=[f"{scheme}://{h}" for h in allowed_hosts for scheme in ("https", "http")],
    )
    manager = StreamableHTTPSessionManager(
        app=create_server(), stateless=True, json_response=False, security_settings=security
    )

    class McpEndpoint:
        # A class instance (not a function) makes Route treat it as a raw ASGI app:
        # exact path, every method, no trailing-slash redirect.
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await manager.handle_request(scope, receive, send)

    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(
        routes=[Route("/mcp", endpoint=McpEndpoint()), Route(HEALTH_PATH, endpoint=healthz, methods=["GET"])],
        middleware=[Middleware(BearerAuthMiddleware, token=token)],
        lifespan=lifespan,
    )


def uvicorn_options(host: str, port: int) -> dict[str, Any]:
    """Proxy headers are trusted only from RSI_FORWARDED_ALLOW_IPS (default: none)."""
    forwarded = _csv_env("RSI_FORWARDED_ALLOW_IPS")
    opts: dict[str, Any] = {"host": host, "port": port, "proxy_headers": bool(forwarded)}
    if forwarded:
        opts["forwarded_allow_ips"] = forwarded
    return opts


def run_http(host: str, port: int) -> None:
    """Open the index (failing loudly), then serve Streamable HTTP until stopped."""
    import uvicorn

    from rsi.config import Settings
    from rsi.indexer.search import SearchEngine

    global _search_engine  # noqa: PLW0603
    token = os.environ.get("RSI_MCP_TOKEN", "").strip()
    if not token:
        raise RuntimeError("RSI_MCP_TOKEN must be set for the http transport")
    settings = Settings.load()
    if not settings.db_path.exists():
        raise RuntimeError(f"no index at {settings.db_path}: run 'rsi index' first")
    engine = SearchEngine(db_path=settings.db_path, settings=settings)
    if engine.store.count() == 0:
        raise RuntimeError(f"index at {settings.db_path} has no rows")
    _search_engine = engine

    app = create_http_app(token=token, allowed_hosts=_csv_env("RSI_MCP_ALLOWED_HOSTS"))
    uvicorn.run(app, **uvicorn_options(host, port))
