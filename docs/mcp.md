# MCP Server Setup

reddit-stash-insights includes an [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server that exposes your Reddit archive as tools for Claude. This lets Claude search your saved content and answer questions about it directly in conversation.

## Prerequisites

1. **Index your archive first** — the MCP server needs an existing search index:
   ```bash
   pip install reddit-stash-insights[search,chat,mcp]
   rsi index ~/reddit
   ```

2. **Have an LLM provider available** (for the `reddit_chat` tool):
   - A GGUF model in `~/.rsi/models/` (auto-detected), or
   - Ollama running locally, or
   - `OPENAI_API_KEY` set

   See [providers.md](providers.md) for setup details. The `reddit_search` tool works without an LLM.

## Setup: Claude Desktop App

### 1. Find the config file

| OS | Path |
|----|------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |

### 2. Find the full path to `rsi`

The Claude desktop app doesn't inherit your shell's PATH, so you need the absolute path:

```bash
which rsi
# e.g., /opt/homebrew/bin/rsi (macOS with Homebrew)
# e.g., /usr/local/bin/rsi (pip install)
# e.g., /home/user/.local/bin/rsi (pip install --user)
```

### 3. Add the MCP server to the config

Edit the config file and add the `mcpServers` section:

```json
{
  "mcpServers": {
    "reddit-stash": {
      "command": "/opt/homebrew/bin/rsi",
      "args": ["mcp"]
    }
  }
}
```

Replace `/opt/homebrew/bin/rsi` with the path from step 2.

If you already have other MCP servers configured, add `reddit-stash` alongside them:

```json
{
  "mcpServers": {
    "existing-server": { "...": "..." },
    "reddit-stash": {
      "command": "/opt/homebrew/bin/rsi",
      "args": ["mcp"]
    }
  }
}
```

### 4. Restart Claude

Fully quit the Claude app (**Cmd+Q** on macOS, not just close the window) and reopen it.

### 5. Verify

Look for the **hammer icon** at the bottom of the chat input. Click it to see the available tools — you should see `reddit_search` and `reddit_chat` listed under "reddit-stash".

## Setup: Claude Code (CLI)

Add to your project's `.claude/settings.json` or global settings:

```json
{
  "mcpServers": {
    "reddit-stash": {
      "command": "rsi",
      "args": ["mcp"]
    }
  }
}
```

Note: Claude Code inherits your shell PATH, so the short `rsi` command works (no absolute path needed).

## Available Tools

### `reddit_search`

Search your saved Reddit content without LLM generation. Returns matching documents with metadata.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *(required)* | Search query text |
| `limit` | integer | 10 | Maximum number of results |
| `mode` | string | `hybrid` | Search mode: `hybrid`, `semantic`, or `keyword` |

### `reddit_chat`

Ask questions about your Reddit archive using RAG. Retrieves relevant documents and generates an answer with an LLM.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | *(required)* | Question about your Reddit archive |
| `limit` | integer | 5 | Number of context documents to retrieve |

## Example Prompts

Once connected, try these in Claude:

- "Search my Reddit archive for posts about machine learning"
- "What cooking recipes have I saved?"
- "Find my saved comments about Python debugging"
- "What subreddits do I engage with most?"
- "Summarize the productivity advice I've saved"
- "Search for posts about home automation with limit 20"

Claude will automatically choose between `reddit_search` and `reddit_chat` based on your question. Direct lookups use search; questions that need synthesis use chat.

## How It Works

```
You ask Claude a question
        │
        ▼
Claude calls reddit_search or reddit_chat
        │
        ▼
rsi MCP server (stdio transport)
        │
        ▼
SearchEngine queries LanceDB index
        │
        ▼
(if reddit_chat) DirectEngine builds RAG prompt → LLM generates answer
        │
        ▼
Results returned to Claude → formatted response to you
```

The MCP server runs as a subprocess managed by Claude. It starts on first use and creates the search engine lazily (the ~2GB embedding model loads on the first tool call, not at startup).

## Troubleshooting

### Tools don't appear in Claude desktop

- **Check the config path** — make sure the JSON is valid (no trailing commas)
- **Use the full path** to `rsi` (run `which rsi` to find it)
- **Fully restart** Claude (Cmd+Q, not just close window)

### "No LLM provider available" error on reddit_chat

The `reddit_chat` tool needs an LLM. Either:
- Drop a GGUF file into `~/.rsi/models/`
- Start Ollama (`ollama serve`)
- Set `OPENAI_API_KEY` in your environment

The `reddit_search` tool works without any LLM.

### "No index found" error

Run `rsi index ~/reddit` first to build the search index.

### Slow first response

The first tool call loads the BGE-M3 embedding model (~2GB). Subsequent calls are fast. If using `llama-cpp`, the GGUF model also loads on first `reddit_chat` call.
