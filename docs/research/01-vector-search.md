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

### Handling long posts (>512 tokens)

- BGE truncates at 512 tokens — for 95% of Reddit posts this is fine
- For the 5% that are longer: truncate to title + first 400 tokens of body
- OR use nomic-embed-text-v1.5 (8192 tokens) as fallback for long posts

## Incremental Indexing Strategy

```
Daily run:
1. reddit-stash generates new markdown files
2. rsi index reads file_log.json for new entries since last run
3. Only new files are embedded and added to LanceDB
4. LanceDB zero-copy append (no rebuild)
```

This means the daily index update takes <10 seconds for ~100 new posts.

## Code Sketch

```python
import lancedb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import yaml

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
db = lancedb.connect("./data/vector_db")

def index_post(md_file: Path) -> dict:
    content = md_file.read_text()
    _, frontmatter, body = content.split("---", 2)
    meta = yaml.safe_load(frontmatter)

    text = f"{meta.get('title', '')}\n\n{body[:2000]}"
    embedding = model.encode(text)

    return {
        "vector": embedding,
        "text": text[:500],  # preview
        "id": meta['id'],
        "subreddit": meta.get('subreddit', ''),
        "score": meta.get('score', 0),
        "timestamp": meta.get('timestamp', ''),
        "file_path": str(md_file)
    }

# Search
def search(query: str, subreddit: str = None, limit: int = 10):
    table = db.open_table("posts")
    q = table.search(model.encode(query)).limit(limit)
    if subreddit:
        q = q.where(f"subreddit = '{subreddit}'")
    return q.to_pandas()
```

## Performance Estimates

| Operation              | 1K posts   | 10K posts  | 50K posts  |
|------------------------|------------|------------|------------|
| Initial index (CPU)    | ~2 min     | ~20 min    | ~90 min    |
| Initial index (GPU)    | ~30 sec    | ~5 min     | ~20 min    |
| Incremental (100 new)  | ~10 sec    | ~10 sec    | ~10 sec    |
| Search latency         | <20ms      | <50ms      | <100ms     |
| Disk usage             | ~5 MB      | ~50 MB     | ~250 MB    |

## Sources

- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- LanceDB docs: https://lancedb.github.io/lancedb/
- BGE models: https://huggingface.co/BAAI/bge-base-en-v1.5
- Firecrawl vector DB comparison 2025: https://www.firecrawl.dev/blog/best-vector-databases-2025
