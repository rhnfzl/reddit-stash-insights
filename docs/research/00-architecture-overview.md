# reddit-stash-insights: Architecture Overview

> Last updated: 2026-02-16. All tech choices independently validated via web search.

## Vision

A companion tool for [reddit-stash](https://github.com/rhnfzl/reddit-stash) that
transforms your Reddit saved content from a passive archive into an active personal
knowledge system with semantic search, knowledge graph exploration, analytics, and
AI-powered Q&A.

## Design Principles

1. **Consume, don't modify**: Reads reddit-stash output (markdown + file_log.json) without changing it
2. **Local-first**: Everything runs on your machine. Cloud LLM is opt-in.
3. **Incremental**: Daily index updates in <10 seconds (no full rebuilds)
4. **Modular**: Install only what you need via optional dependency groups
5. **No abandoned dependencies**: Every core technology is actively maintained with large community backing

## Architecture

```
reddit-stash output directory
    |
    |  (markdown files + file_log.json)
    v
┌─────────────────────────────────────────────────┐
│              reddit-stash-insights               │
│                                                  │
│  ┌──────────┐                                    │
│  │  Parser   │  Read markdown + YAML frontmatter │
│  │  (core)   │  Read file_log.json               │
│  └────┬──────┘                                   │
│       │                                          │
│       ├────────────────┬─────────────────┐       │
│       v                v                 v       │
│  ┌──────────┐   ┌────────────┐    ┌──────────┐  │
│  │ Indexer   │   │   Graph    │    │ Analytics│  │
│  │          │   │  Builder   │    │          │  │
│  │ BGE-M3 + │   │ NetworkX + │    │ Pandas + │  │
│  │ LanceDB  │   │ BERTopic + │    │ Plotly   │  │
│  │          │   │ SQLite     │    │          │  │
│  └────┬─────┘   └─────┬──────┘    └────┬─────┘  │
│       │               │               │         │
│       └───────┬───────┴───────┬───────┘         │
│               v               v                  │
│        ┌────────────┐  ┌──────────────┐          │
│        │  Search /  │  │  Streamlit   │          │
│        │  Chat /    │  │  Dashboard   │          │
│        │  MCP       │  │              │          │
│        └────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────┘
```

## Module Structure

```
reddit-stash-insights/
├── src/rsi/
│   ├── __init__.py
│   ├── cli.py                  # Typer CLI entry point
│   ├── config.py               # Settings / configuration
│   │
│   ├── core/                   # Always installed
│   │   ├── __init__.py
│   │   ├── parser.py           # Markdown + frontmatter parser
│   │   ├── models.py           # Dataclasses: Post, Comment, Subreddit
│   │   └── file_log.py         # Read file_log.json
│   │
│   ├── indexer/                # pip install rsi[search]
│   │   ├── __init__.py
│   │   ├── embedder.py         # BGE-M3 embedding (dense+sparse hybrid)
│   │   ├── vector_store.py     # LanceDB operations
│   │   └── search.py           # Semantic + keyword hybrid search
│   │
│   ├── graph/                  # pip install rsi[graph]
│   │   ├── __init__.py
│   │   ├── builder.py          # NetworkX graph construction
│   │   ├── topics.py           # BERTopic topic extraction
│   │   ├── analysis.py         # Centrality, communities, bridges
│   │   ├── persistence.py      # SQLite graph storage
│   │   └── visualization.py    # Pyvis export
│   │
│   ├── analytics/              # pip install rsi[analytics]
│   │   ├── __init__.py
│   │   ├── metrics.py          # Stats computation
│   │   ├── trends.py           # Temporal analysis
│   │   ├── profiler.py         # Interest profiling
│   │   └── reports.py          # Generate reports
│   │
│   ├── chat/                   # pip install rsi[chat]
│   │   ├── __init__.py
│   │   ├── rag.py              # RAG pipeline (LlamaIndex)
│   │   ├── llm_provider.py     # llama-cpp-python / Ollama / Cloud APIs
│   │   └── prompts.py          # System prompts for Q&A
│   │
│   ├── mcp/                    # pip install rsi[mcp]
│   │   ├── __init__.py
│   │   └── server.py           # MCP server (SDK v1.26+)
│   │
│   └── ui/                     # pip install rsi[ui]
│       ├── __init__.py
│       ├── app.py              # Streamlit dashboard
│       ├── pages/
│       │   ├── search.py
│       │   ├── graph.py
│       │   ├── analytics.py
│       │   └── chat.py
│       └── components/
│           ├── graph_viewer.py
│           └── charts.py
│
├── tests/
│   ├── test_parser.py
│   ├── test_indexer.py
│   ├── test_graph.py
│   ├── test_analytics.py
│   └── test_mcp.py
│
├── docs/
│   ├── research/               # This research directory
│   └── plans/                  # Implementation plans
│
├── pyproject.toml
├── README.md
├── LICENSE
└── .gitignore
```

## Tech Stack (Validated 2026-02-16)

| Component | Technology | Why This Over Alternatives | Status |
|-----------|-----------|---------------------------|--------|
| **Embeddings** | **BGE-M3** (BAAI, 550M params, 1024d) | #1 open-source on MTEB for retrieval (63.0). Unique: supports dense + sparse + ColBERT in ONE model = native hybrid search. 8192 token context. Beats BGE-base-en-v1.5 which was only dense. | Active, BAAI maintained |
| **Vector DB** | **LanceDB** (v0.29+) | Embedded (no server), zero-copy incremental updates (critical for daily syncs), built-in hybrid search, Lance columnar format. Latest: Jan 2026 newsletter, HNSW-accelerated indexing. | Active, v0.29.2 (Feb 9, 2026) |
| **Graph** | **NetworkX** (v3.x) | 14K stars, maintained since 2004. For 1K-50K nodes, no graph DB needed. 1000+ built-in algorithms. Pure Python. | Active, very low risk |
| **Graph Persistence** | **SQLite** | Python stdlib. Nodes + edges in relational tables, reload into NetworkX. | Zero risk |
| **Topic Modeling** | **BERTopic** (v0.16+) | 99.5% coherence on Reddit short-text data (arXiv:2412.14486). Best for social media content. Supports zero-shot topics. No serious challenger for short text. | Active, 6.2K stars |
| **RAG Framework** | **LlamaIndex** (v0.11+) | Built for document RAG. Native markdown + LanceDB support. Simpler than LangChain for retrieval Q&A. IBM and multiple 2026 comparisons confirm it's superior for pure RAG. | Active, well-funded |
| **Local LLM (Option A)** | **llama-cpp-python** | Direct llama.cpp bindings. Zero overhead (no Ollama server). Supports both embeddings AND inference. GGUF models from HuggingFace. Best for minimal dependency footprint. | Active, ggml-org maintained |
| **Local LLM (Option B)** | **Ollama** | Simpler UX (one-line install). Better for users who want `ollama pull` convenience. 10-15% overhead vs raw llama.cpp but zero config. | Active, very popular |
| **Cloud LLM** | **Claude API / OpenAI / Google** | Gemini 1.5 Flash: cheapest ($0.08/1M input, free tier 1500 req/day). GPT-4o mini: best quality/price. Claude Haiku 4.5: 200K context. | Active |
| **Visualization** | **Pyvis + Plotly** | Pyvis: NetworkX → interactive HTML graph. Plotly: interactive charts for analytics. Both proven, both active. | Active |
| **Dashboard** | **Streamlit** (v1.35+) | Fastest Python→web path. Built-in chat UI, widgets, session state. Snowflake-backed. Confirmed best for this use case vs Gradio (ML-demo focused) and Panel (steeper learning curve). | Active |
| **MCP** | **mcp Python SDK** (v1.26+) | Official Anthropic SDK. FastMCP incorporated. Works with Claude Desktop, Cursor, OpenAI Agents SDK. | Active, v1.26.0 (Jan 2026) |
| **CLI** | **Typer** | Built on Click but uses Python type hints. Automatic help generation, less boilerplate. Modern alternative to raw Click. | Active, by tiangolo (FastAPI creator) |

### Recommended Local LLM Models (Feb 2026 Benchmarks)

| Model | Params | RAM | Best For | Notes |
|-------|--------|-----|----------|-------|
| **Phi-4** | 14B | 10GB | Best quality/size ratio | Matches many 70B models on reasoning |
| **Qwen 2.5** | 7B | 8GB | General RAG + multilingual | Strong all-rounder |
| **SmolLM3** | 3B | 4GB | Minimal hardware | Outperforms Llama-3.2-3B at same size |
| **Gemma-3n-E2B** | 5B (2B active) | 4GB | On-device / low resource | Selective parameter activation |
| **Qwen 2.5 Coder** | 7B | 8GB | Code-heavy subreddits | Beats GPT-4o on coding benchmarks |

> Note: The earlier recommendation of "Llama 3.1 8B = GPT-3.5 quality" was outdated.
> Current small models (Phi-4, Qwen 2.5) compete with GPT-4-level quality on many tasks.

### llama.cpp vs Ollama Decision

| Criteria | llama-cpp-python | Ollama |
|----------|-----------------|--------|
| Setup | `pip install llama-cpp-python` | `brew install ollama` |
| Dependencies | Just the Python package | Separate binary + server |
| Overhead | Zero (direct C++ bindings) | 10-15% (HTTP API layer) |
| Embeddings | Built-in (same binary) | Separate `ollama embed` |
| Model management | Manual GGUF download | `ollama pull model` |
| Context window | Full control (up to 32K+) | Default ~11K, configurable |
| Best for | Minimal deps, max performance | Easy setup, beginners |

**Recommendation**: Support BOTH. Default to Ollama for UX, offer llama-cpp-python
for users who want zero-overhead or can't install Ollama.

### REJECTED Technologies

| Technology | Reason | Source |
|-----------|--------|--------|
| KuzuDB | Acquired by Apple Oct 2025, GitHub archived | [The Register](https://www.theregister.com/2025/10/14/kuzudb_abandoned/) |
| BGE-base-en-v1.5 | Superseded by BGE-M3 (hybrid retrieval, higher MTEB) | [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| all-MiniLM-L6-v2 | Outdated (2021), MTEB 56.3 vs BGE-M3's 63.0 | [BentoML comparison](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models) |
| ChromaDB | No zero-copy incremental updates | [Firecrawl comparison](https://www.firecrawl.dev/blog/best-vector-databases-2025) |
| FAISS | No metadata storage, no incremental updates, no hybrid search | - |
| LangChain (as sole framework) | Over-engineered for document RAG; LlamaIndex better for retrieval | [IBM comparison](https://www.ibm.com/think/topics/llamaindex-vs-langchain) |
| Neo4j | Requires JVM server, 52x slower ingestion vs embedded options | - |
| Click (raw) | Typer is built on Click but with less boilerplate via type hints | [Typer docs](https://typer.tiangolo.com/alternatives/) |

## Dependency Groups

```toml
[project]
name = "reddit-stash-insights"
requires-python = ">=3.10"
dependencies = [
    "pyyaml>=6.0",
    "typer>=0.12",
]

[project.optional-dependencies]
search = [
    "sentence-transformers>=3.0",
    "lancedb>=0.29",
    "FlagEmbedding>=1.2",   # For BGE-M3 hybrid retrieval
]
graph = [
    "networkx>=3.0",
    "bertopic>=0.16",
    "pyvis>=0.3",
    "sentence-transformers>=3.0",
]
analytics = [
    "plotly>=5.0",
    "pandas>=2.0",
]
chat = [
    "llama-index>=0.11",
]
chat-local = [
    "llama-cpp-python>=0.3",      # Direct llama.cpp (no Ollama needed)
]
chat-ollama = [
    "ollama>=0.3",                 # Ollama client
]
mcp = [
    "mcp>=1.26",
]
ui = [
    "streamlit>=1.35",
    "plotly>=5.0",
]
all = [
    "reddit-stash-insights[search,graph,analytics,chat,chat-ollama,mcp,ui]",
]
```

## CLI Commands

```bash
# Core
rsi index /path/to/reddit/           # Build/update vector + graph indexes
rsi search "Rust async"              # Semantic search
rsi search --subreddit python "web"  # Filtered search

# Graph
rsi graph build                      # Build knowledge graph
rsi graph export graph.html          # Export interactive Pyvis HTML
rsi graph topics                     # Show discovered topics
rsi graph bridges                    # Show knowledge bridge posts

# Analytics
rsi analytics                        # Print summary stats
rsi analytics --report report.html   # Generate HTML report
rsi analytics --trends               # Show interest trends

# Chat (RAG)
rsi chat                             # Interactive chat with saved posts
rsi chat --provider openai           # Use cloud LLM
rsi chat --provider llama-cpp        # Use llama.cpp directly
rsi chat --question "What..."        # One-shot question

# MCP
rsi mcp                              # Start MCP server (for Claude Desktop)

# Dashboard
rsi serve                            # Start Streamlit dashboard
```

## Implementation Phases

### Phase 1: Foundation (Core + Search)
- Markdown parser + data models
- file_log.json reader
- BGE-M3 embedding + LanceDB indexing
- Semantic + hybrid search CLI
- Basic tests

### Phase 2: Knowledge Graph
- NetworkX graph construction
- BERTopic topic extraction
- Community detection + centrality analysis
- Pyvis visualization export
- SQLite persistence

### Phase 3: Analytics
- Core metrics computation (saves over time, top subreddits, content types)
- Interest profiling (TF-IDF weighted by recency)
- Trend detection (monthly topic evolution)
- Plotly charts + HTML report generation

### Phase 4: Chat (RAG)
- LlamaIndex RAG pipeline
- LLM provider abstraction: llama-cpp-python / Ollama / Cloud APIs
- Interactive CLI chat
- One-shot question mode

### Phase 5: MCP Server
- MCP tool definitions (search, analytics, similar posts, chat)
- Claude Desktop / Cursor integration
- Configurable via environment variables

### Phase 6: Dashboard
- Streamlit app with multi-page layout
- Search page (semantic + filters)
- Graph viewer page (embedded Pyvis)
- Analytics page (Plotly charts)
- Chat page (conversational RAG)

## Input Data Contract

reddit-stash-insights reads these files from the reddit-stash output directory:

### 1. Markdown Files (*.md)

```
reddit/
├── {subreddit}/
│   ├── POST_{id}.md
│   ├── COMMENT_{id}.md
│   ├── SAVED_POST_{id}.md
│   ├── SAVED_COMMENT_{id}.md
│   ├── UPVOTE_POST_{id}.md
│   └── UPVOTE_COMMENT_{id}.md
```

Each file has YAML frontmatter:
```yaml
---
id: 1njd7m5
subreddit: /r/fitbod
timestamp: 2025-09-17 13:29:49
author: /u/complexrexton
comments: 3
permalink: https://reddit.com/r/fitbod/comments/1njd7m5/...
---
```

### 2. file_log.json

```json
{
  "1njd7m5-fitbod-Submission-POST": {
    "subreddit": "fitbod",
    "type": "Submission",
    "file_path": "fitbod/POST_1njd7m5.md"
  }
}
```

### 3. Media Files (optional)

Images and videos alongside markdown files. Not indexed but can be referenced.

## Existing Open-Source Landscape

**No mature project exists for this exact use case.** Closest matches:
- raphaelsty/knowledge (~500 stars): GitHub/HN bookmarks → Neo4j graph (not Reddit)
- reddit-research-mcp (86 stars): Hosted service (not local-first)
- Reor (8.5K stars): Private AI knowledge management (not Reddit-specific)

**reddit-stash-insights would be the first** open-source
"Reddit saved posts → local vector DB → knowledge graph → semantic search → RAG chat → MCP server" tool.

## Sources

- MTEB Leaderboard 2026: [Ailog](https://app.ailog.fr/en/blog/guides/choosing-embedding-models), [BentoML](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models)
- LanceDB status: [Jan 2026 newsletter](https://lancedb.com/blog/newsletter-january-2026/), [PyPI](https://pypi.org/project/lancedb/)
- KuzuDB death: [The Register](https://www.theregister.com/2025/10/14/kuzudb_abandoned/), [MacDailyNews](https://macdailynews.com/2026/02/12/apple-acquires-graph-database-maker-kuzu/)
- LlamaIndex vs LangChain 2026: [IBM](https://www.ibm.com/think/topics/llamaindex-vs-langchain), [Leon Consulting](https://leonstaff.com/blogs/langchain-vs-llamaindex-rag-wars/)
- Small LLMs 2026: [BentoML SLMs](https://www.bentoml.com/blog/the-best-open-source-small-language-models), [DataCamp](https://www.datacamp.com/blog/top-small-language-models)
- llama.cpp vs Ollama: [Openxcell](https://www.openxcell.com/blog/llama-cpp-vs-ollama/), [DecodesFuture](https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide)
- MCP SDK: [GitHub](https://github.com/modelcontextprotocol/python-sdk), [v1.26.0](https://github.com/modelcontextprotocol/python-sdk/releases)
- BERTopic on Reddit: [arXiv:2412.14486](https://arxiv.org/html/2412.14486v1)
- Typer: [Typer docs](https://typer.tiangolo.com/)
