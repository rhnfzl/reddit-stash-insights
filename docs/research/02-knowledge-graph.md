# Knowledge Graph Research

## Problem Statement

Users want to explore connections in their saved Reddit history:
- Which subreddits share topics?
- What are the "knowledge bridges" connecting different interest areas?
- How do topics cluster and evolve over time?

## Graph Database Selection

### IMPORTANT: KuzuDB is DEAD

KuzuDB was acquired by Apple in October 2025. The entire GitHub organization was
archived on Oct 10, 2025. Do NOT use Kuzu — it will never receive updates.

Source: https://www.theregister.com/2025/10/14/kuzudb_abandoned/
Source: https://macdailynews.com/2026/02/12/apple-acquires-graph-database-maker-kuzu/

### Recommended: NetworkX + SQLite (No Graph DB Needed)

For our scale (1K-50K nodes), a dedicated graph database is overkill.

**NetworkX** (14K+ GitHub stars, actively maintained):
- Pure Python, zero infrastructure
- 1000+ built-in graph algorithms (community detection, centrality, shortest paths, PageRank)
- In-memory, fast for our scale
- Perfect Pyvis integration for visualization
- Serializable via pickle, JSON, GraphML, GEXF

**SQLite** for persistence:
- Store nodes and edges in relational tables
- Reload into NetworkX on startup
- Familiar, zero-risk technology

**Why this beats a graph DB for our use case**:
- No abandoned-project risk (NetworkX has been maintained since 2004)
- No server, no Docker, no subprocess
- Python-native — all manipulation is just Python code
- For 50K nodes, in-memory is fine (~50MB RAM)

### Alternatives Considered

| Option | Status | Verdict |
|--------|--------|---------|
| ~~KuzuDB~~ | Acquired by Apple, archived | DEAD |
| FalkorDBLite | New (Nov 2025), spawns subprocess | Too new, not truly embedded |
| DuckDB + DuckPGQ | Community extension, under development | Promising but immature for graph |
| Neo4j Community | Requires JVM server | Overkill for personal tool |
| NetworkX | 20+ years maintained, 14K stars | Perfect for our scale |

## Graph Schema

### Nodes

| Node Type | Properties | Source |
|-----------|-----------|--------|
| Post | id, title, url, score, created_utc, file_path | Markdown frontmatter |
| Subreddit | name | Folder name / frontmatter |
| Author | username | Frontmatter |
| Topic | name, keywords | BERTopic auto-extraction |

### Edges

| Edge Type | From | To | Properties |
|-----------|------|-----|-----------|
| POSTED_IN | Post | Subreddit | - |
| AUTHORED_BY | Post | Author | - |
| HAS_TOPIC | Post | Topic | probability |
| SIMILAR_TO | Post | Post | cosine_similarity |
| CO_SAVED_WITH | Subreddit | Subreddit | count (same-week saves) |

### Advanced Analysis Patterns

**1. Topic Clusters via Community Detection**
```python
import networkx as nx
from networkx.algorithms.community import louvain_communities

communities = louvain_communities(G)
# Reveals: {"Programming": [r/rust, r/python, r/golang],
#           "ML/AI": [r/MachineLearning, r/LocalLLaMA]}
```

**2. Knowledge Bridges (High Betweenness Centrality)**
```python
centrality = nx.betweenness_centrality(G)
bridges = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
# Posts that connect different topic clusters
# e.g., "Rust for ML" bridges Programming and ML clusters
```

**3. Interest Evolution Over Time**
```python
# Monthly subreddit save counts -> trend analysis
df['month'] = pd.to_datetime(df['created_utc']).dt.to_period('M')
trends = df.groupby(['month', 'subreddit']).size().unstack(fill_value=0)
```

## Topic Extraction

### Recommended: BERTopic

- **GitHub**: https://github.com/MaartenGr/BERTopic (6.2K stars)
- **Coherence on Reddit**: 99.5% (vs 73.3% for LDA, per arXiv:2412.14486)
- **No LLM needed**: Uses sentence embeddings + HDBSCAN clustering + c-TF-IDF
- **Zero-shot mode**: Pre-define topics, auto-assign posts
- **Hierarchical**: Parent-child topic relationships

```python
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
topic_model = BERTopic(embedding_model=embedding_model, min_topic_size=5)
topics, probs = topic_model.fit_transform(post_texts, embeddings)
```

### Alternative: Subreddit as Topic Proxy

- Zero compute
- Use both: subreddit as coarse category, BERTopic for fine-grained topics

## Visualization

### Recommended: Pyvis

- **GitHub**: https://github.com/WestHealth/pyvis (1.1K stars)
- Interactive HTML output (drag, zoom, hover tooltips)
- Physics simulation (nodes spring together naturally)
- Direct NetworkX integration: `net.from_nx(G)`
- Streamlit-embeddable

```python
from pyvis.network import Network

net = Network(height="800px", width="100%", bgcolor="#1a1a2e", font_color="white")
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.show("reddit_graph.html")
```

### Node Styling

| Node Type | Color | Size |
|-----------|-------|------|
| Post | Blue (#4361ee) | Proportional to score |
| Subreddit | Green (#2ec4b6) | Proportional to post count |
| Author | Orange (#ff6b35) | Proportional to saves from them |
| Topic | Purple (#7209b7) | Proportional to post count |

## SQLite Persistence Schema

```sql
CREATE TABLE nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,  -- post, subreddit, author, topic
    label TEXT,
    properties JSON
);

CREATE TABLE edges (
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    type TEXT NOT NULL,  -- posted_in, authored_by, has_topic, similar_to
    weight REAL DEFAULT 1.0,
    properties JSON,
    FOREIGN KEY (source_id) REFERENCES nodes(id),
    FOREIGN KEY (target_id) REFERENCES nodes(id)
);

CREATE INDEX idx_edges_source ON edges(source_id);
CREATE INDEX idx_edges_target ON edges(target_id);
CREATE INDEX idx_nodes_type ON nodes(type);
```

## Sources

- KuzuDB death: https://www.theregister.com/2025/10/14/kuzudb_abandoned/
- Apple acquires Kuzu: https://macdailynews.com/2026/02/12/apple-acquires-graph-database-maker-kuzu/
- BERTopic Reddit benchmarks: https://arxiv.org/html/2412.14486v1
- DuckPGQ: https://duckdb.org/docs/stable/guides/sql_features/graph_queries.html
- FalkorDBLite: https://www.falkordb.com/blog/falkordblite-embedded-python-graph-database/
- NetworkX: https://networkx.org/
- Pyvis: https://github.com/WestHealth/pyvis
