# Vector Search & Semantic Search Research

## Problem Statement

Reddit Stash generates markdown files with YAML frontmatter organized by subreddit.
Users need semantic search ("find posts about Rust async") beyond keyword matching.

## Embedding Model Selection

> Updated 2026-02-16 based on current MTEB leaderboard data.

### Recommended: BGE-M3 (BAAI)

- **Parameters**: 550M
- **Dimensions**: 1024
- **Context window**: 8192 tokens (handles even long Reddit threads)
- **MTEB score**: 63.0 (highest open-source for retrieval, Jan 2026)
- **Unique capability**: Supports dense + sparse + ColBERT retrieval IN ONE MODEL
  - Dense: traditional semantic similarity
  - Sparse: lexical matching (like BM25 but learned)
  - ColBERT: token-level interaction for fine-grained matching
- **Install**: `pip install FlagEmbedding` or `pip install sentence-transformers`
- **Memory**: ~2-3 GB

This is the consensus #1 open-source embedding model for RAG as of early 2026,
confirmed by multiple independent sources (KDnuggets, Ailog MTEB analysis, BentoML,
AIMResearch benchmark on 490K Amazon reviews).

### MTEB Leaderboard (Jan 2026)

| Rank | Model | MTEB Score | Dimensions | Open Source? |
|------|-------|-----------|-----------|-------------|
| 1 | Gemini-embedding-001 | 68.32 | 3072 | No (API) |
| 2 | Qwen3-Embedding-8B | 70.58 | 4096 | Yes (but 8B = heavy) |
| 3 | Voyage-3-large | 66.8 | 1536 | No (API) |
| 4 | Cohere embed-v4 | 65.2 | 1024 | No (API) |
| 5 | OpenAI text-3-large | 64.6 | 3072 | No (API) |
| **6** | **BGE-M3** | **63.0** | **1024** | **Yes (free)** |
| 7 | Nomic-embed-text-v1.5 | 59.4 | 768 | Yes |
| 8 | all-MiniLM-L6-v2 | 56.3 | 384 | Yes |

**Why BGE-M3 over Qwen3-Embedding-8B**: Qwen3 scores higher but is 8B params
(~16GB RAM for inference). BGE-M3 at 550M is 15x smaller and still #1 among
practical open-source models. For a personal tool running on laptops, BGE-M3 is
the right tradeoff.

### Why NOT BGE-base-en-v1.5 (Previous Recommendation)

- Only supports dense retrieval (no hybrid search)
- 512 token context (BGE-M3 has 8192)
- Lower MTEB score
- Superseded by BGE-M3 from the same team (BAAI)

### Why NOT all-MiniLM-L6-v2

- Outdated (2021 model)
- MTEB 56.3 vs BGE-M3's 63.0 (12% lower)
- 384 dimensions, 256 token limit

## Vector Database Selection

### Recommended: LanceDB

- **Architecture**: Embedded (like SQLite), no server needed
- **Key feature**: Zero-copy incremental updates (perfect for daily GitHub Actions runs)
- **Format**: `.lance` files on disk
- **Hybrid search**: Vector similarity + metadata filtering in one query
- **Scale**: Sub-100ms queries on 1M vectors
- **Production users**: Midjourney (1B+ vectors)
- **Install**: `pip install lancedb`
- **Status (Feb 2026)**: Actively maintained, Apache 2.0 license

### Why NOT ChromaDB

ChromaDB is great for prototyping but LanceDB wins for our use case because:
- Zero-copy incremental updates (ChromaDB requires full re-add)
- Better performance at scale (LanceDB uses Lance columnar format)
- Built-in hybrid search (ChromaDB's is basic)
- Both are embedded/serverless

### Why NOT FAISS

- No metadata storage (just vectors)
- No incremental updates (must rebuild index)
- No hybrid search
- Great for billion-scale, overkill complexity for our use case

### Comparison Matrix

| Feature           | LanceDB    | ChromaDB   | FAISS      | Qdrant     |
|-------------------|------------|------------|------------|------------|
| No Docker         | Yes        | Yes        | Yes        | No         |
| Incremental       | Zero-copy  | Re-add     | Manual     | Yes        |
| Hybrid Search     | Built-in   | Basic      | No         | Advanced   |
| Ideal Scale       | 1M-100M    | <500K      | Any        | 1M-100M    |
| Setup Complexity  | 1 line     | 1 line     | 10 lines   | Docker     |
| Metadata Storage  | Yes        | Yes        | No         | Yes        |

## Chunking Strategy

### Recommended: Post-Level Chunking (1 file = 1 vector)

For each markdown file, concatenate:
- Title
- Self-text / post body
- Top 3-5 comments (they're often why you saved it)

Store as metadata: id, subreddit, score, timestamp, file_path, author.

### Why NOT comment-level chunking

- Over-fragments the data (too many tiny vectors)
- Comments lack context without the parent post
- Adds complexity without meaningful accuracy gains for search

### Handling long posts

BGE-M3 supports 8192 tokens natively, so even very long Reddit threads are handled
without truncation. No fallback model needed (unlike BGE-base-en-v1.5's 512 limit).

## Hybrid Search: BM25 + Semantic + Sparse

### Why Hybrid Search Matters for Reddit Content

Pure semantic search misses exact keyword matches:
- Query: "PRAW" → semantic search might return "Reddit API library" posts but miss ones mentioning "PRAW" by name
- Query: "ELI5 quantum computing" → BM25 catches the exact "ELI5" keyword, semantic catches conceptually similar explanations

IBM research confirms that **three-way hybrid retrieval (BM25 + dense + sparse)
is the optimal configuration for RAG**, outperforming any two-way combination.

### Architecture: Three-Way Hybrid in LanceDB

LanceDB provides ALL THREE search modes natively — no external BM25 library needed:

```
                    User Query
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   ┌─────────┐   ┌───────────┐   ┌───────────┐
   │  BM25   │   │   Dense   │   │  Sparse   │
   │ (Tantivy│   │  Vectors  │   │  Vectors  │
   │  FTS)   │   │ (BGE-M3)  │   │ (BGE-M3)  │
   └────┬────┘   └─────┬─────┘   └─────┬─────┘
        │               │               │
        └───────┬───────┴───────┬───────┘
                ▼               ▼
         ┌────────────────────────┐
         │   Reranker (RRF or    │
         │  LinearCombination)   │
         └───────────┬───────────┘
                     ▼
              Top-K Results → RAG Prompt
```

**Component 1: BM25 via Tantivy (LanceDB built-in)**
- Exact keyword matching using BM25 scoring
- Powered by Tantivy (Rust-based FTS engine)
- Created via `table.create_fts_index("text")`
- Catches: exact terms, acronyms (PRAW, ELI5), proper nouns

**Component 2: Dense Vectors (BGE-M3)**
- 1024-dimensional semantic embeddings
- Captures meaning regardless of exact words
- Catches: paraphrases, conceptually similar content

**Component 3: Sparse Vectors (BGE-M3)**
- Learned term importance weights (like BM25 but trained)
- Generated alongside dense vectors at zero extra cost
- Catches: important terms weighted by context, not just frequency

### LanceDB Rerankers

LanceDB has built-in rerankers to merge results from multiple search modes:

| Reranker | How It Works | Speed | Quality |
|----------|-------------|-------|---------|
| `LinearCombinationReranker` | Weighted score: 0.7 × vector + 0.3 × FTS (default) | Fastest | Good |
| `RRFReranker` | Reciprocal Rank Fusion: combines rank positions | Fast | Better |
| `CrossEncoderReranker` | Neural model re-scores pairs | Slow | Best |
| `CohereReranker` | Cohere API re-ranking | Medium | Very Good |
| Custom | Implement `Reranker` base class | Varies | Varies |

**Recommended**: Start with `RRFReranker(k=60)` — good quality, no API cost, fast.
Upgrade to `CrossEncoderReranker` if precision matters more than speed.

### Code Sketch: Hybrid Search Pipeline

```python
import lancedb
from FlagEmbedding import BGEM3FlagModel
from lancedb.rerankers import RRFReranker
from pathlib import Path
import yaml

# Initialize BGE-M3 (generates dense + sparse in one call)
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
db = lancedb.connect("./data/vector_db")

def index_post(md_file: Path) -> dict:
    content = md_file.read_text()
    _, frontmatter, body = content.split("---", 2)
    meta = yaml.safe_load(frontmatter)

    text = f"{meta.get('title', '')}\n\n{body[:4000]}"

    # BGE-M3 returns dense + sparse + colbert in one call
    output = model.encode(
        [text],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False  # ColBERT is optional, increases storage
    )

    return {
        "vector": output["dense_vecs"][0],       # 1024-dim dense
        "sparse_vector": output["lexical_weights"][0],  # learned sparse
        "text": text,                              # full text for BM25 FTS
        "id": meta["id"],
        "subreddit": meta.get("subreddit", ""),
        "score": meta.get("score", 0),
        "timestamp": meta.get("timestamp", ""),
        "file_path": str(md_file),
    }

# Create table with FTS index
table = db.create_table("posts", data=posts)
table.create_fts_index("text")  # Enables BM25 via Tantivy

# Hybrid search: BM25 + dense vectors, merged by RRF
def hybrid_search(query: str, subreddit: str = None, limit: int = 10):
    reranker = RRFReranker(k=60)

    q = table.search(query, query_type="hybrid") \
             .rerank(reranker=reranker) \
             .limit(limit)

    if subreddit:
        q = q.where(f"subreddit = '{subreddit}'")

    return q.to_pandas()

# Pure semantic search (when you want only meaning-based results)
def semantic_search(query: str, limit: int = 10):
    query_embedding = model.encode([query], return_dense=True)["dense_vecs"][0]
    return table.search(query_embedding).limit(limit).to_pandas()

# Pure BM25 search (when you want exact keyword matching)
def keyword_search(query: str, limit: int = 10):
    return table.search(query, query_type="fts").limit(limit).to_pandas()
```

### When Each Search Mode Wins

| Query Type | Best Mode | Example |
|------------|-----------|---------|
| Exact terms, acronyms | BM25 | "PRAW", "ELI5", "CUDA 12.4" |
| Conceptual / paraphrased | Dense semantic | "how to learn programming" |
| Technical + contextual | Hybrid (all three) | "Rust async error handling patterns" |
| Known post title | BM25 | "Fitbod Screen Stuck" |
| Vague / exploratory | Dense semantic | "interesting machine learning stuff" |

### RAG Pipeline with Hybrid Search

The hybrid search feeds directly into the RAG pipeline:

```python
def rag_query(question: str, llm_provider) -> str:
    # 1. Hybrid search for relevant posts
    results = hybrid_search(question, limit=5)

    # 2. Build context from top results
    context = "\n\n---\n\n".join([
        f"[r/{row['subreddit']}] {row['text'][:1000]}"
        for _, row in results.iterrows()
    ])

    # 3. Generate answer grounded in retrieved posts
    prompt = f"""Based on the following saved Reddit posts, answer the question.
Cite which post(s) you used.

Posts:
{context}

Question: {question}

Answer:"""

    return llm_provider.generate(prompt)
```

### BM25 Implementation Options Considered

| Option | How | Verdict |
|--------|-----|---------|
| **LanceDB Tantivy FTS** | `table.create_fts_index("text")` | **Recommended** — native, no extra deps |
| rank_bm25 (Python) | Separate in-memory BM25 | Needs manual text preprocessing, redundant |
| BM25S (scipy-based) | 500x faster than rank_bm25 | Only needed if >1M docs, overkill for us |
| Elasticsearch | Full BM25 server | Way overkill, requires JVM |

**LanceDB's Tantivy-based FTS is the clear winner** — it gives us BM25 search
natively within the same database we already use for vectors, with built-in reranking.
No separate library, no index sync issues, no extra process.

### Tantivy FTS Limitations (Current)

- Only available in Python SDK (not TypeScript)
- FTS index stored on local filesystem only (not object storage yet)
- No incremental FTS indexing yet (full rebuild on new data — fast for <100K docs)
- LanceDB team is actively working on Rust-level FTS integration

For our scale (1K-50K posts), these limitations are not blockers.

## Incremental Indexing Strategy

```
Daily run:
1. reddit-stash generates new markdown files
2. rsi index reads file_log.json for new entries since last run
3. Only new files get BGE-M3 embeddings (dense + sparse)
4. LanceDB zero-copy append for vectors
5. Rebuild FTS index (fast for <100K posts, ~seconds)
```

## Performance Estimates

| Operation              | 1K posts   | 10K posts  | 50K posts  |
|------------------------|------------|------------|------------|
| Initial index (CPU)    | ~3 min     | ~30 min    | ~120 min   |
| Initial index (GPU)    | ~45 sec    | ~7 min     | ~30 min    |
| Incremental (100 new)  | ~15 sec    | ~15 sec    | ~15 sec    |
| Hybrid search latency  | <30ms      | <60ms      | <120ms     |
| Disk usage (vectors)   | ~8 MB      | ~80 MB     | ~400 MB    |
| Disk usage (FTS index) | ~2 MB      | ~20 MB     | ~100 MB    |

Note: BGE-M3 (550M params) is larger than BGE-base-en-v1.5 (110M), so indexing
is somewhat slower but quality is significantly better.

## Sources

- MTEB Leaderboard: [HuggingFace](https://huggingface.co/spaces/mteb/leaderboard)
- BGE-M3 model card: [HuggingFace](https://huggingface.co/BAAI/bge-m3)
- LanceDB hybrid search docs: [LanceDB](https://docs.lancedb.com/search/hybrid-search)
- LanceDB FTS with Tantivy: [LanceDB](https://docs.lancedb.com/search/full-text-search)
- LanceDB rerankers: [LanceDB blog](https://lancedb.com/blog/hybrid-search-and-custom-reranking-with-lancedb-4c10a6a3447e/)
- IBM Blended RAG paper: [Infiniflow analysis](https://infiniflow.org/blog/best-hybrid-search-solution)
- BM25 vs sparse vs hybrid: [Medium](https://medium.com/@dewasheesh.rana/bm25-vs-sparse-vs-hybrid-search-in-rag-from-layman-to-pro-e34ff21c4ada)
- BM25S (fast BM25): [HuggingFace blog](https://huggingface.co/blog/xhluca/bm25s)
- Firecrawl vector DB comparison 2025: [Firecrawl](https://www.firecrawl.dev/blog/best-vector-databases-2025)
