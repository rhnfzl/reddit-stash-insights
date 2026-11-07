# Passage indexing result

Long threads retain late evidence in bounded passages. Search returns the best passage from each source thread.
The implementation is ready for the coordinator's development-set comparison. It makes no retrieval-quality claim yet.

## Files and interfaces

- Added `src/rsi/indexer/chunker.py` and `tests/test_chunker.py`.
- Added `src/rsi/core/identity.py`; evaluation re-exports its shared `canonical_doc_id` import.
- Updated indexing only in `src/rsi/cli.py`, plus `config.py`, embedder, search, vector store, and index metadata.
- Added chunk, metadata, retry, stale-row, tokenizer-adapter, and bounded-search regressions.
- Added tests to nox and compatibility CI. Updated configuration reference and README migration notes.
- No parser/model/scanner changes were necessary. No commits were made by this worker.

`iter_chunks(item, *, encode, decode, max_tokens=512, overlap=64)` yields dictionaries lazily.
The active embedder exposes `encode(text)` and `decode(tokens)`.
Local BGE uses its actual model tokenizer without special tokens. OpenAI uses its existing tiktoken encoder.
The project installation exposes the local tokenizer at `.venv/lib/python3.14/site-packages/FlagEmbedding/inference/embedder/encoder_only/m3.py`.

Passages repeat bounded title/comment context. They split at comment and paragraph boundaries.
Oversized code and comments use token-bounded windows. Character slicing keeps Unicode sequences intact.
There is no document character cap. `RSI_MAX_TEXT_CHARS` remains a deprecated compatibility field and is ignored by indexing.

Rows preserve original `id`, `file_path`, and `content_type`.
`doc_id` and `source_doc_id` identify the archive document. `chunk_id` is source identity plus ordinal.
`evidence_id` identifies the post or embedded comment. `parent_doc_id` is the canonical source thread.
`parent_id` records an available immediate comment parent. Missing relationships stay empty.
`permalink` identifies passage evidence; `source_permalink` preserves the original document link.
Offsets `source_start` and `source_end` address `item.body`, identified by `source_offset_basis = "body"`.
The companion writer's nested blockquotes determine embedded comment boundaries and parent relationships.

`SearchEngine.index_items(items)` streams passage generation and embedding batches.
It holds at most `index_chunk_size` passage dictionaries and one vector batch.
The existing scanner still materializes parsed documents. This change does not promise streaming archive parsing.
A content generation hash supports stale-row removal after that source finishes embedding.
Stable chunk upserts preserve sibling passages and make retries idempotent.
The low-level unchunked `index_texts` helper requires a fresh directory.

Search shares canonical identity with evaluation. It preserves first occurrence. Candidate depths are `max(50, limit)`, `max(100, 2 * limit)`, and `max(200, 4 * limit)`.
Each mode returns unique threads, including standalone comments repeated inside a post.
Duplicate-heavy candidate pools can still produce fewer results than requested. The requested limit must be a positive integer.

## Schema and migration

Schema 2 metadata includes `chunks = {version: 1, max_tokens: 512, overlap: 64}`.
Readers reject unknown schemas, changed chunk settings, changed embedding identities, and incomplete builds.
Schema 1 and metadata-free BGE indexes remain readable. Writers require a fresh directory or explicit `--rebuild`.
The CLI sets `complete = false` before embedding and true only after the full-text index finishes.
A failed embedding batch leaves previous stale rows available for retry; readers refuse the incomplete build.
All rebuild tests use disposable temporary directories. No live homelab index or corpus was changed.

Configuration supports `[index].chunk_tokens`, `[index].chunk_overlap`, `RSI_CHUNK_TOKENS`, and `RSI_CHUNK_OVERLAP`.
Defaults are 512 and 64. Overlap must be nonnegative and smaller than the token limit.

## Verification

New missing-module chunk tests failed before implementation. Duplicate-thread search tests failed before implementation.
Title-only and legacy-write regressions also failed before their fixes.

The final project-runtime command passes 129 tests with eight optional/model skips:

```bash
RSI_SKIP_SLOW_TESTS=1 .venv/bin/python -m unittest tests.test_chunker tests.test_models tests.test_parser tests.test_scanner tests.test_config tests.test_index_build tests.test_vector_store tests.test_search tests.test_embedder tests.test_openai_embedder tests.test_cli_search tests.test_evaluation tests.test_cli_eval -q
.venv/bin/ruff check src/rsi/ tests/
.venv/bin/mypy src/rsi/
git diff --check
```

Ruff passes. Mypy passes all 30 source files. Diff whitespace checks pass.
A project tiktoken self-check produces 113 Unicode/code passages, each at most 512 tokens, and retains late evidence.
The verified tokenizer path is `.venv/lib/python3.14/site-packages/tiktoken/__init__.py`.
An aggregate-only scan of the private corpus reports 1,055 posts, 1,298 comments, and zero parse errors.
No private questions, labels, or held-out scores were read. No new models were downloaded or loaded.

The coordinator owns development metric deltas, BGE experiments, and release acceptance.
Heavy-model tests and live deployment behavior remain unchecked by this worker.

## Review corrections

Two confirmed review findings are fixed. Both regression groups fail before their fixes and pass afterward.

Oversized chunk settings previously allowed text to remain in storage after its embedding input was truncated.
The CLI now rejects chunk settings above the active encoder capacity before metadata writes or rebuild deletion.
`index_items` repeats this validation for direct callers. Metadata records the content capacity in `input_token_capacity`.
OpenAI exposes its configured `max_input_tokens`. Fake embedders used by index tests must expose this positive integer too.

The project runtime is FlagEmbedding 1.4.2. Its resolved source is
`/Users/rehan-8v/projects/personal/reddit-stash-insights/.venv/lib/python3.14/site-packages/FlagEmbedding/inference/embedder/encoder_only/m3.py`.
Its default `passage_max_length` is 512 tokens, including tokenizer special tokens.
Local encoding now explicitly uses the smaller usable tokenizer/model position limit.
For BGE, the tokenizer limit of 8,192 and two special tokens expose 8,190 content tokens.
This prevents the previous hidden truncation of two tokens from a full 512-token passage.
The model weights stay unchanged. Comparisons against the frozen baseline also include removal of this hidden encoder truncation.

The fixed 200-candidate search ceiling previously returned only 200 results when 300 unique threads were requested.
Candidate depths now scale with the caller limit, while the default depth sequence stays 50, 100, 200.
The regression returns all 300 requested unique threads. Invalid limits fail before querying.

The focused review-fix suite passes 86 tests with eight optional/model skips.
The expanded chunk/core/search/evaluation suite then passes 135 tests with eight optional/model skips.
Global Ruff and mypy checks pass after these fixes. No model downloads or private relevance scoring occurred.
