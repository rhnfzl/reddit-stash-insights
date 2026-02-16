"""MCP server exposing reddit-stash-insights as tools for Claude Code."""
from __future__ import annotations

import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from rsi.chat.engine import DirectEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton engine
# ---------------------------------------------------------------------------

_engine: DirectEngine | None = None


def _get_engine() -> DirectEngine:
    """Return (or create) the shared :class:`DirectEngine`."""
    global _engine  # noqa: PLW0603
    if _engine is None:
        from rsi.chat.providers.base import create_provider
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine, SearchMode

        settings = Settings.load()
        search_engine = SearchEngine(db_path=settings.db_path)
        llm = create_provider(provider=settings.llm_provider, model=settings.llm_model)
        _engine = DirectEngine(
            search_engine=search_engine,
            llm=llm,
            search_mode=SearchMode(settings.chat_search_mode),
            max_history_turns=settings.chat_max_history,
        )
    return _engine


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
                "description": "Maximum number of results to return",
                "default": 10,
            },
            "mode": {
                "type": "string",
                "description": "Search mode",
                "enum": ["hybrid", "semantic", "keyword"],
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
                "description": "Number of context documents to retrieve",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


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
    """Handle the ``reddit_search`` tool call."""
    from rsi.indexer.search import SearchMode

    query: str = arguments["query"]
    limit: int = arguments.get("limit", 10)
    mode_str: str = arguments.get("mode", "hybrid")

    engine = _get_engine()
    results = engine._search.search(query, limit=limit, mode=SearchMode(mode_str))
    text = _format_search_results(results)
    return [TextContent(type="text", text=text)]


async def _handle_reddit_chat(arguments: dict[str, Any]) -> list[TextContent]:
    """Handle the ``reddit_chat`` tool call."""
    query: str = arguments["query"]
    limit: int = arguments.get("limit", 5)

    engine = _get_engine()
    resp = engine.chat(query, limit=limit)

    # Build source list
    source_lines: list[str] = []
    for i, src in enumerate(resp.sources, 1):
        sub = src.get("subreddit", "?")
        file_path = src.get("file_path", "?")
        source_lines.append(f"  [{i}] r/{sub} -- {file_path}")

    parts = [resp.answer]
    if source_lines:
        parts.append("\nSources:")
        parts.extend(source_lines)

    return [TextContent(type="text", text="\n".join(parts))]


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
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
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
