# Reddit Stash Insights

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CI](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/rhnfzl/reddit-stash-insights/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

> Companion to [reddit-stash](https://github.com/rhnfzl/reddit-stash) — adds semantic search and AI chat to your Reddit archive.

## How It Works

**reddit-stash** saves your Reddit content as markdown files. **reddit-stash-insights** indexes those files and lets you search and chat with them:

```
reddit-stash saves  -->  rsi indexes  -->  search & chat
   (markdown)          (BGE-M3 + LanceDB)    (RAG + LLM)
```

## Features

- **Hybrid search** — combines semantic (BGE-M3, 1024-dim) and keyword (BM25 via Tantivy) search with RRF reranking
- **RAG chat** — ask questions about your archive using local or cloud LLMs with streaming responses
- **Multiple LLM providers** — llama-cpp (local GGUF), Ollama, OpenAI-compatible (vLLM, LM Studio)
- **Auto-fallback** — automatically detects available providers and falls back gracefully
- **MCP server** — expose search and chat as tools for Claude Code
- **Streamlit UI** — web-based chat interface with source citations
- **S3 support** — fetch reddit-stash data directly from S3 buckets
- **CLI-first** — every feature is accessible via the `rsi` command

## Quick Start

```bash
pip install reddit-stash-insights[search]
rsi index ~/reddit                   # Build search index (first run downloads ~2GB model)
rsi search "best python libraries"   # Search your archive
```

To add AI chat:

```bash
pip install reddit-stash-insights[search,chat]
rsi chat "what are my most saved topics?"   # Single question
rsi chat                                     # Interactive REPL
```

## Installation

```bash
pip install reddit-stash-insights           # Core only (scan)
```

Install extras for additional features:

| Extra | What it adds | Key dependencies |
|-------|-------------|-----------------|
| `search` | Semantic + keyword search | sentence-transformers, LanceDB, FlagEmbedding |
| `chat` | Local LLM inference | llama-cpp-python |
| `chat-ollama` | Ollama LLM provider | ollama |
| `chat-openai` | OpenAI-compatible provider | openai |
| `mcp` | MCP server for Claude Code | mcp |
| `ui` | Streamlit chat interface | streamlit, plotly |
| `all` | Everything above | — |

```bash
# Common combinations
pip install reddit-stash-insights[search,chat]       # Search + local LLM
pip install reddit-stash-insights[all]                # Everything
pip install reddit-stash-insights[search,chat-openai] # Search + OpenAI/vLLM
```

## Usage

### Scan

Inspect a reddit-stash directory without indexing:

```bash
rsi scan ~/reddit
# Found 1523 posts and 847 comments
```

### Index

Build or update the vector search index:

```bash
rsi index ~/reddit
rsi index ~/reddit --s3-bucket my-bucket  # Fetch from S3 first, then index
```

### Search

Query your indexed content with three search modes:

```bash
rsi search "machine learning papers"              # Hybrid (default)
rsi search "transformers" --mode semantic          # Semantic only
rsi search "python tutorial" --mode keyword        # BM25 keyword only
rsi search "rust" --subreddit programming -n 5     # Filter by subreddit
```

### Chat

Ask questions using RAG with your preferred LLM:

```bash
# Single question (streams response by default)
rsi chat "summarize my saved cooking recipes"

# Interactive REPL with conversation history
rsi chat
# you> what topics do I save most?
# you> /sources    (show source documents)
# you> /clear      (reset history)
# you> /quit

# Override provider for a session
rsi chat --provider ollama --model qwen2.5:7b
rsi chat --provider openai --model gpt-4o
```

### Streamlit UI

Launch the web-based chat interface:

```bash
pip install reddit-stash-insights[search,chat,ui]
streamlit run src/rsi/ui/chat.py
```

### MCP Server

Expose your Reddit archive as tools for Claude. Works with both the Claude desktop app and Claude Code.

```bash
pip install reddit-stash-insights[search,chat,mcp]
```

This exposes two tools:

| Tool | Description |
|------|-------------|
| `reddit_search` | Search your archive (no LLM needed for results) |
| `reddit_chat` | Ask questions with RAG — retrieves context and generates answers |

See [docs/mcp.md](docs/mcp.md) for setup instructions (Claude desktop app, Claude Code) and usage examples.

## Configuration

Settings live in `~/.rsi/config.toml`. Minimal example:

```toml
[llm]
provider = "llama-cpp"
model = "~/.rsi/models/qwen2.5-7b-instruct-q4_k_m.gguf"

[chat]
context_docs = 5
search_mode = "hybrid"
```

Environment variables override config file values (e.g., `RSI_LLM_PROVIDER`, `RSI_S3_BUCKET`).

See [docs/configuration.md](docs/configuration.md) for the full reference.

## LLM Providers

| Provider | Type | Best for |
|----------|------|----------|
| `llama-cpp` | Local GGUF | Privacy, offline use, no API costs |
| `ollama` | Local HTTP | Easy model management, multiple models |
| `openai` | Cloud/API | Highest quality, also works with vLLM/LM Studio |

The system automatically detects available providers and falls back in order: `llama-cpp` -> `ollama` -> `openai`.

See [docs/providers.md](docs/providers.md) for setup guides.

## Development

```bash
pip install -e ".[all,dev]"

# Tests (unittest, not pytest)
python -m unittest tests.test_models tests.test_parser tests.test_scanner tests.test_cli tests.test_config -v  # Core
python -m unittest tests.test_chat_providers tests.test_chat_engine tests.test_cli_chat -v                     # Chat (mocked)
python -m unittest tests.test_embedder tests.test_vector_store tests.test_search tests.test_cli_search -v      # Search (slow, loads model)

# Lint
ruff check src/ tests/

# Multi-version testing
nox              # All sessions (3.11-3.14)
nox -s core      # Core tests only
```

## Architecture

The system is organized in layers: **Core** (parsing, scanning) -> **Indexer** (embedding, vector store, search) -> **Chat** (RAG engine, LLM providers) -> **UI/MCP** (Streamlit, MCP server). Each layer depends only on the ones below it.

See [docs/architecture.md](docs/architecture.md) for the full system design.

## License

[MIT](LICENSE)
