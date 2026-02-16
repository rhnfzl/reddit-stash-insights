# Architecture

reddit-stash-insights is organized in four layers, each depending only on the layers below it.

## Layer Diagram

```
┌──────────────────────────────────────────────────┐
│  UI / Integration                                │
│  ┌──────────────────┐  ┌───────────────────────┐ │
│  │  Streamlit Chat   │  │  MCP Server (stdio)   │ │
│  │  (ui/chat.py)     │  │  (mcp_server.py)      │ │
│  └────────┬─────────┘  └───────────┬───────────┘ │
├───────────┴────────────────────────┴─────────────┤
│  Chat / RAG                                      │
│  ┌──────────────────────────────────────────────┐│
│  │  DirectEngine                                ││
│  │  (chat/engine.py)                            ││
│  │  Orchestrates: search → prompt → LLM → answer││
│  └──────────┬───────────────────────┬───────────┘│
│  ┌──────────┴──────────┐ ┌──────────┴──────────┐ │
│  │  Prompt Builder     │ │  LLM Providers      │ │
│  │  (chat/prompt.py)   │ │  llama-cpp | ollama  │ │
│  │                     │ │  openai              │ │
│  └─────────────────────┘ └─────────────────────┘ │
├──────────────────────────────────────────────────┤
│  Indexer / Search                                │
│  ┌──────────────────────────────────────────────┐│
│  │  SearchEngine                                ││
│  │  (indexer/search.py)                         ││
│  │  Modes: hybrid | semantic | keyword          ││
│  └──────────┬───────────────────────┬───────────┘│
│  ┌──────────┴──────────┐ ┌──────────┴──────────┐ │
│  │  Embedder           │ │  VectorStore         │ │
│  │  BGE-M3 (1024-dim)  │ │  LanceDB + Tantivy  │ │
│  │  Dense + Sparse     │ │  FTS (BM25)          │ │
│  └─────────────────────┘ └─────────────────────┘ │
├──────────────────────────────────────────────────┤
│  Core                                            │
│  ┌────────────┐ ┌──────────┐ ┌────────────────┐  │
│  │  Scanner   │ │  Parser  │ │  S3 Fetch      │  │
│  │  Walks dir │ │  MD+YAML │ │  (optional)    │  │
│  └────────────┘ └──────────┘ └────────────────┘  │
│  ┌────────────┐ ┌───────────────────────────────┐ │
│  │  Models    │ │  Config (TOML + env vars)     │ │
│  └────────────┘ └───────────────────────────────┘ │
└──────────────────────────────────────────────────┘
```

## Data Flow

```
Markdown files (from reddit-stash)
        │
        ▼
    Scanner ─── walks directory, finds POST_*.md / COMMENT_*.md
        │
        ▼
    Parser ─── extracts YAML frontmatter + body → RedditPost / RedditComment
        │
        ▼
    Embedder ─── BGE-M3 model → 1024-dim dense + sparse vectors
        │
        ▼
    VectorStore ─── stores in LanceDB, creates Tantivy FTS index
        │
        ▼
    SearchEngine ─── hybrid (RRF), semantic, or keyword search
        │
        ▼
    DirectEngine ─── retrieves context → builds prompt → calls LLM
        │
        ▼
    LLM Provider ─── generates answer (streaming or complete)
        │
        ▼
    Response ─── answer + source citations
```

## Component Reference

| Module | Layer | Responsibility | Key File |
|--------|-------|---------------|----------|
| `core/scanner.py` | Core | Walk directory tree, find markdown files | `ScanResult` (posts + comments + errors) |
| `core/parser.py` | Core | Parse YAML frontmatter + markdown body | `parse_post()`, `parse_comment()` |
| `core/models.py` | Core | Data models with `search_text()` method | `RedditPost`, `RedditComment` |
| `core/s3_fetch.py` | Core | Download reddit-stash files from S3 | `fetch_from_s3()` |
| `config.py` | Core | TOML + env var config loading | `Settings.load()` |
| `indexer/embedder.py` | Indexer | BGE-M3 wrapper, dense + sparse embedding | `Embedder.embed()`, `embed_query()` |
| `indexer/vector_store.py` | Indexer | LanceDB storage + Tantivy FTS | `VectorStore.hybrid_search()` |
| `indexer/search.py` | Indexer | Orchestrate embedder + store | `SearchEngine.search()` |
| `chat/engine.py` | Chat | RAG orchestration | `DirectEngine.chat()`, `chat_stream()` |
| `chat/prompt.py` | Chat | System prompt + context formatting | `build_messages()` |
| `chat/history.py` | Chat | Conversation memory with eviction | `ChatHistory` |
| `chat/providers/base.py` | Chat | `LLMProvider` Protocol + factory | `create_provider()` |
| `chat/providers/availability.py` | Chat | Pre-flight checks + fallback | `find_available_provider()` |
| `chat/providers/llama_cpp_provider.py` | Chat | Local GGUF inference | `LlamaCppProvider` |
| `chat/providers/ollama_provider.py` | Chat | Ollama HTTP client | `OllamaProvider` |
| `chat/providers/openai_provider.py` | Chat | OpenAI-compatible API | `OpenAIProvider` |
| `ui/chat.py` | UI | Streamlit chat interface | Streamlit app |
| `mcp_server.py` | Integration | MCP tools for Claude Code | `reddit_search`, `reddit_chat` |
| `cli.py` | CLI | Typer CLI entry point | `scan`, `index`, `search`, `chat`, `mcp` |

## Key Design Patterns

### LLM Provider Protocol

All LLM backends implement the `LLMProvider` Protocol (defined in `chat/providers/base.py`):

```python
@runtime_checkable
class LLMProvider(Protocol):
    @property
    def context_window(self) -> int: ...
    def generate(self, messages: list[ChatMessage]) -> str: ...
    def stream(self, messages: list[ChatMessage]) -> Iterator[str]: ...
```

This uses Python's structural typing — any class with these methods satisfies the protocol, no inheritance required. The `create_provider()` factory instantiates the right class by name.

### Deferred Imports

Optional dependencies (llama-cpp-python, ollama, openai, streamlit, mcp) are imported at the point of use, not at module level. This means the core package works with just `pyyaml` and `typer`, and heavier dependencies are only loaded when needed.

### Hybrid Search (RRF)

Hybrid search combines results from two sources using Reciprocal Rank Fusion:

1. **Semantic** — BGE-M3 dense vector similarity via LanceDB
2. **Keyword** — BM25 full-text search via Tantivy (built into LanceDB)

RRF merges the two ranked lists without requiring score normalization, which is robust when the two scoring systems have different scales.

### Lazy Singleton Engine (MCP)

The MCP server creates the `DirectEngine` (embedding model + LLM) on first tool call, not at server startup. This avoids loading ~2GB of model weights until actually needed.
