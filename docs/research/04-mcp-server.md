# MCP Server Research

## Problem Statement

Users want to query their saved Reddit posts directly from Claude Desktop, Cursor,
or any MCP-compatible AI tool using natural language.

## What is MCP?

Model Context Protocol (MCP) is an open standard by Anthropic (Nov 2024) for AI
models to connect to external tools and data sources. Think "USB-C for AI apps" —
one server works with all MCP clients.

## Architecture

```
Claude Desktop / Cursor / Cline
    |
    | (MCP protocol over stdio)
    v
reddit-stash-insights MCP Server
    |
    +-- search_saved_posts(query, subreddit, limit)
    +-- get_post_details(post_id)
    +-- get_analytics(metric)
    +-- get_similar_posts(post_id, limit)
    +-- list_subreddits()
    +-- get_topic_graph(subreddit)
    +-- chat_with_posts(question)  [RAG]
```

## MCP Tool Schema

### Tool 1: search_saved_posts
```json
{
  "name": "search_saved_posts",
  "description": "Semantic search across saved Reddit posts",
  "parameters": {
    "query": {"type": "string", "description": "Search query"},
    "subreddit": {"type": "string", "description": "Filter by subreddit (optional)"},
    "limit": {"type": "integer", "default": 5, "description": "Max results"},
    "after": {"type": "string", "description": "Only posts after this date (YYYY-MM-DD)"},
    "before": {"type": "string", "description": "Only posts before this date (YYYY-MM-DD)"}
  }
}
```

### Tool 2: get_post_details
```json
{
  "name": "get_post_details",
  "description": "Get full content of a saved post by ID",
  "parameters": {
    "post_id": {"type": "string", "description": "Reddit post ID"}
  }
}
```

### Tool 3: get_analytics
```json
{
  "name": "get_analytics",
  "description": "Get analytics about saved posts",
  "parameters": {
    "metric": {
      "type": "string",
      "enum": ["top_subreddits", "saves_over_time", "top_authors",
               "content_types", "interest_profile", "topic_trends"]
    },
    "period": {"type": "string", "description": "Time period (e.g., '30d', '6m', '1y')"}
  }
}
```

### Tool 4: get_similar_posts
```json
{
  "name": "get_similar_posts",
  "description": "Find posts similar to a given post",
  "parameters": {
    "post_id": {"type": "string"},
    "limit": {"type": "integer", "default": 5}
  }
}
```

### Tool 5: chat_with_posts (RAG)
```json
{
  "name": "chat_with_posts",
  "description": "Ask questions about saved posts using AI. Retrieves relevant posts and generates an answer.",
  "parameters": {
    "question": {"type": "string", "description": "Question about your saved content"}
  }
}
```

## Implementation

### Python MCP SDK

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("reddit-stash-insights")

@server.tool()
async def search_saved_posts(query: str, subreddit: str = None, limit: int = 5):
    """Semantic search across saved Reddit posts."""
    results = vector_search(query, subreddit=subreddit, limit=limit)
    return [{"title": r.title, "subreddit": r.subreddit,
             "url": r.permalink, "preview": r.text[:200]} for r in results]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write)
```

### Claude Desktop Config

```json
{
  "mcpServers": {
    "reddit-stash-insights": {
      "command": "python",
      "args": ["-m", "rsi.mcp"],
      "env": {
        "RSI_DATA_DIR": "/path/to/reddit-stash/output"
      }
    }
  }
}
```

## Example Conversations

**User**: "What posts did I save about Rust async programming?"
**Claude**: *calls search_saved_posts(query="Rust async programming")*
→ Returns 5 relevant posts with titles, subreddits, and previews

**User**: "Summarize my interests based on saved posts"
**Claude**: *calls get_analytics(metric="interest_profile")*
→ Returns topic distribution: "40% Programming, 25% ML/AI, ..."

**User**: "Find posts similar to that Tokio tutorial I saved"
**Claude**: *calls get_similar_posts(post_id="1abc...")*
→ Returns 5 semantically similar posts

## Existing MCP Servers for Reference

- **mcp-server-sqlite**: SQLite queries via MCP (reference for DB-backed servers)
- **mcp-server-filesystem**: File access via MCP (reference for file-based servers)
- No existing MCP server for personal knowledge bases found

## Sources

- MCP spec: https://modelcontextprotocol.io/
- Python MCP SDK: https://github.com/modelcontextprotocol/python-sdk
- MCP server examples: https://github.com/modelcontextprotocol/servers
