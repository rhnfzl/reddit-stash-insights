# Configuration Reference

reddit-stash-insights loads settings from three sources, in order of precedence:

1. **Environment variables** (highest priority)
2. **Config file** (`~/.rsi/config.toml`)
3. **Built-in defaults**

## Config File

The default location is `~/.rsi/config.toml`. All sections and keys are optional — defaults are sensible for most setups.

```toml
# ~/.rsi/config.toml

[embedding]
model = "BAAI/bge-m3"       # HuggingFace model ID for embeddings
use_fp16 = true              # Use FP16 for faster inference (recommended)

[store]
db_path = "~/.rsi/index/vectors.lance"   # LanceDB vector database location

[llm]
provider = "llama-cpp"       # LLM backend: "llama-cpp", "ollama", or "openai"
model = "qwen2.5:7b"         # Model name, tag, or path to GGUF file

[chat]
context_docs = 5             # Number of retrieved documents per RAG query
search_mode = "hybrid"       # Default search mode: "hybrid", "semantic", "keyword"
max_history = 10             # Max conversation turns to keep in REPL/UI

[s3]
bucket = "my-reddit-bucket"  # S3 bucket containing reddit-stash data
prefix = "reddit/"           # Key prefix for reddit-stash files
cache_dir = "~/.rsi/cache/s3"  # Local cache for downloaded S3 files
```

## Environment Variable Overrides

Each config key has a corresponding environment variable. These take the highest precedence.

| Environment Variable | Config Key | Default |
|---------------------|-----------|---------|
| `RSI_EMBEDDING_MODEL` | `embedding.model` | `BAAI/bge-m3` |
| `RSI_DB_PATH` | `store.db_path` | `~/.rsi/index/vectors.lance` |
| `RSI_LLM_PROVIDER` | `llm.provider` | `llama-cpp` |
| `RSI_LLM_MODEL` | `llm.model` | `qwen2.5:7b` |
| `RSI_CHAT_CONTEXT_DOCS` | `chat.context_docs` | `5` |
| `RSI_CHAT_SEARCH_MODE` | `chat.search_mode` | `hybrid` |
| `RSI_S3_BUCKET` | `s3.bucket` | *(none)* |
| `RSI_S3_PREFIX` | `s3.prefix` | `reddit/` |

**Example:**

```bash
# Override the LLM provider for a single command
RSI_LLM_PROVIDER=ollama rsi chat "what did I save recently?"

# Or export for the session
export RSI_S3_BUCKET=my-reddit-archive
rsi index ~/reddit --s3-bucket $RSI_S3_BUCKET
```

## Section Details

### `[embedding]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | string | `BAAI/bge-m3` | HuggingFace model ID. BGE-M3 produces 1024-dim dense + sparse vectors for hybrid search. The model (~2GB) downloads automatically on first use. |
| `use_fp16` | bool | `true` | Use FP16 precision. Faster and uses less memory with negligible quality loss. |

### `[store]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | path | `~/.rsi/index/vectors.lance` | Path to the LanceDB database directory. Supports both absolute and `~`-prefixed paths. |

### `[llm]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | `llama-cpp` | LLM backend to use. See [providers.md](providers.md) for setup details. |
| `model` | string | `qwen2.5:7b` | Interpretation depends on provider: a GGUF file path for `llama-cpp`, a model tag for `ollama`, or a model name for `openai`. |

### `[chat]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `context_docs` | int | `5` | Number of documents retrieved from the index for each RAG query. Higher values provide more context but increase token usage. |
| `search_mode` | string | `hybrid` | Default search mode for RAG retrieval. Options: `hybrid` (semantic + BM25 with RRF), `semantic` (vector only), `keyword` (BM25 only). |
| `max_history` | int | `10` | Maximum conversation turns retained in the REPL and Streamlit UI. Older turns are evicted when this limit is reached. |

### `[s3]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `bucket` | string | *(none)* | S3 bucket name. When set, `rsi index --s3-bucket` fetches files from S3 before indexing. |
| `prefix` | string | `reddit/` | Key prefix within the bucket where reddit-stash files are stored. |
| `cache_dir` | path | `~/.rsi/cache/s3` | Local directory for caching downloaded S3 files. |

S3 uses standard AWS credential resolution (environment variables, `~/.aws/credentials`, IAM roles). Custom endpoints (MinIO, LocalStack) can be configured via the `AWS_ENDPOINT_URL` environment variable.

## CLI Overrides

Most settings can also be overridden per-command via CLI flags:

```bash
rsi index ~/reddit --db-path /tmp/test.lance
rsi search "query" --mode semantic --limit 20
rsi chat --provider ollama --model qwen2.5:7b
rsi chat --no-stream  # Disable streaming (print complete response)
```

CLI flags take the highest precedence, above environment variables and config file.

## Data Directory

All rsi data lives under `~/.rsi/`:

```
~/.rsi/
├── config.toml          # Configuration file
├── index/
│   └── vectors.lance/   # LanceDB vector database
├── models/              # GGUF model files (for llama-cpp)
│   └── *.gguf
└── cache/
    └── s3/              # Cached S3 downloads
```
