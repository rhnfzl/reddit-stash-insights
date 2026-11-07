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
provider = "local"           # "local" (BGE-M3 in process) or "openai" (OpenAI-compatible API)
model = "BAAI/bge-m3"       # HuggingFace model ID, or API model name for "openai"
use_fp16 = true              # Use FP16 for faster inference (recommended)
# base_url = "http://localhost:4000/v1"  # "openai" only; defaults to OPENAI_BASE_URL

[index]
chunk_tokens = 512          # Maximum embedding tokens per passage, including context
chunk_overlap = 64          # Overlap inside oversized paragraphs

[store]
db_path = "~/.rsi/index/vectors.lance"   # LanceDB vector database location

[llm]
provider = "llama-cpp"       # LLM backend: "llama-cpp", "ollama", or "openai"
model = "qwen2.5:7b"         # Model name, tag, or path to GGUF file
allow_cloud_fallback = false # Permit an OpenAI-compatible fallback only after explicit consent

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
| `RSI_EMBEDDING_PROVIDER` | `embedding.provider` | `local` |
| `RSI_EMBEDDING_MODEL` | `embedding.model` | `BAAI/bge-m3` (`text-embedding-3-small` for `openai`) |
| `RSI_EMBEDDING_BASE_URL` | `embedding.base_url` | `OPENAI_BASE_URL`, else the OpenAI API |
| `RSI_EMBEDDING_API_KEY` | *(env only)* | `OPENAI_API_KEY` |
| `RSI_EMBEDDING_BATCH` | *(env only)* | `128` inputs per embeddings request |
| `RSI_INDEX_CHUNK` | *(env only)* | `512` items embedded and inserted at a time |
| `RSI_CHUNK_TOKENS` | `index.chunk_tokens` | `512` embedding tokens per passage |
| `RSI_CHUNK_OVERLAP` | `index.chunk_overlap` | `64` tokens of overlap inside oversized paragraphs |
| `RSI_MAX_TEXT_CHARS` | Deprecated compatibility setting | Ignored by passage indexing; complete documents are indexed |
| `RSI_DB_PATH` | `store.db_path` | `~/.rsi/index/vectors.lance` |
| `RSI_LLM_PROVIDER` | `llm.provider` | `llama-cpp` |
| `RSI_LLM_MODEL` | `llm.model` | `qwen2.5:7b` |
| `RSI_ALLOW_CLOUD_FALLBACK` | `llm.allow_cloud_fallback` | `false` (accepts only `true` or `false`) |
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

### Embedding providers

`local` needs the `search` extra (torch, BGE-M3). `openai` needs the `embed-openai` extra and calls any
OpenAI-compatible `/v1/embeddings` endpoint: each input is cut to 8000 tokens with the model tokenizer
(tiktoken), requests stay under 128 inputs and 250k tokens, and the SDK retries rate limits and server errors.
The indexer packs adjacent short comments into passages and preserves their original body text.
Each passage includes `evidence_refs`, a nonempty list of evidence IDs, permalinks, parent IDs, and clipped body offsets.
Packed passages use the source thread as their primary evidence and URL. Single-evidence passages retain their evidence-specific fields.
Oversized paragraphs and code
use overlapping windows with the active embedding tokenizer. Unicode characters stay intact.
The 512-token limit includes repeated title and comment context. The indexer retains evidence after
character 32,000 and writes at most `RSI_INDEX_CHUNK` passage vectors per batch.

Search returns one best passage per source thread. For the default result limit, it requests 50, 100, then 200 candidates.
For larger limits, candidate depths are at least the requested limit, twice that limit, then four times that limit.
Duplicate-heavy rankings can still return fewer source threads than requested. Search limits must be positive integers.
Rows retain the original `id`, `file_path`, and `content_type`. `chunk_id` identifies the passage;
`source_doc_id` identifies its archive document; `evidence_id` identifies the post or embedded comment;
`parent_doc_id` identifies the source thread. `permalink` links to the passage evidence when available.
`source_permalink` retains the document URL. Missing links stay empty. `parent_id` records the immediate
comment parent when available. `source_start` and `source_end` are character offsets into the parsed
body, as recorded by `source_offset_basis = "body"`.

`rsi index` writes schema 2 metadata in `rsi_index.json`. Metadata includes provider, model, dimension,
row count, parse errors, build time, `input_token_capacity`, and `chunks: {version: 3, max_tokens: 512, overlap: 64}`.
Readers reject unknown schemas, different chunk settings, and different embedding models before querying.
The `complete` marker is false during indexing and true only after the full-text index finishes.
Readers refuse unfinished builds. Retry the same index command to finish an interrupted build.
Passage upserts use `chunk_id`. Successful updates remove stale passages from shorter documents.
A failed embedding batch does not remove the previous document's remaining passages.

Existing schema 1 indexes and indexes without metadata remain readable. They retain their original
whole-document retrieval. Indexes without metadata use `local`/`BAAI/bge-m3` identity.
Writers refuse these legacy indexes to prevent mixed schemas. Build a separate directory:

```bash
rsi index ~/reddit --db-path ~/.rsi/index/passages.lance
RSI_DB_PATH=~/.rsi/index/passages.lance rsi health
```

Point readers at the new directory after validation. Keep the old directory for rollback.
`--rebuild` deletes the selected table, so use it only when discarding that index is intended.
Changing chunk settings or the embedding provider requires a fresh directory or an explicit rebuild.
The unreleased chunk algorithms 1 and 2 require a fresh directory or rebuild before algorithm 3 writes.
Legacy schema 1 remains readable.
The passage token limit cannot exceed the active encoder capacity. Invalid limits are rejected before metadata writes or rebuild deletion.
Local capacity excludes tokenizer special tokens. Local encoding explicitly uses the model/tokenizer sequence limit.
OpenAI capacity uses the configured per-input token cap.
The internal local embedder disables unused BGE sparse output. Retrieval uses dense vectors and full-text search.
Direct `Embedder()` callers retain sparse output by default; `return_sparse=False` is an explicit keyword option.
`RSI_INDEX_CHUNK` must be positive. Chunk overlap must be nonnegative and smaller than the token limit.

### Index and health exit codes

| Code | `rsi index` | `rsi health` / `rsi search` |
|------|-------------|-----------------------------|
| 0 | Indexed (or nothing to index without `--max-error-pct`) | Healthy / searched |
| 1 | Unexpected error (for example the embeddings API failed after retries) | Unhealthy: no index, unfinished build, no rows, or the probe search failed |
| 2 | Bad arguments | Bad arguments |
| 3 | `--max-error-pct` gate: zero items or too many parse errors; no metadata written | |
| 4 | Legacy schema or incompatible chunk/model settings; use a fresh directory or `--rebuild` | Unknown schema or incompatible chunk/model settings |

`rsi health [--db-path PATH] [--mode keyword|hybrid|semantic]` searches for words taken from stored rows,
longest first (up to five tries), so stop words and over-long tokens do not fail a valid index.
The default `keyword` mode makes no embedding call, so it works as a container healthcheck during an
embeddings API outage.

### `[store]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `db_path` | path | `~/.rsi/index/vectors.lance` | Path to the LanceDB database directory. Supports both absolute and `~`-prefixed paths. |

### `[llm]`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `provider` | string | `llama-cpp` | LLM backend to use. See [providers.md](providers.md) for setup details. |
| `model` | string | `qwen2.5:7b` | Interpretation depends on provider: a GGUF file path for `llama-cpp`, a model tag for `ollama`, or a model name for `openai`. |
| `allow_cloud_fallback` | bool | `false` | Permit an OpenAI-compatible fallback when local providers fail. Explicit `openai` selection permits that endpoint without this setting. |

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
