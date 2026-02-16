# Tech Stack Validation Log

> Each technology was independently verified via web search on 2026-02-16.
> This log records what was checked, what changed, and why.

## 1. Embedding Model

**Original recommendation**: BGE-base-en-v1.5 (768d)
**Updated to**: BGE-M3 (1024d, 550M params)

**What changed**: BGE-base-en-v1.5 is superseded by BGE-M3 from the same BAAI team.
BGE-M3 uniquely supports dense + sparse + ColBERT retrieval in one model, enabling
native hybrid search without separate BM25. MTEB score 63.0 vs BGE-base-en-v1.5's
lower score. Context window expanded from 512 to 8192 tokens.

**Current MTEB top open-source** (Jan 2026):
1. Qwen3-Embedding-8B: 70.58 (but 8B params = too heavy for personal tool)
2. BGE-M3: 63.0 (sweet spot: quality + practical size)

**Sources**: [Ailog MTEB analysis](https://app.ailog.fr/en/blog/guides/choosing-embedding-models), [BentoML guide](https://www.bentoml.com/blog/a-guide-to-open-source-embedding-models), [KDnuggets top 5](https://www.kdnuggets.com/top-5-embedding-models-for-your-rag-pipeline), [AIMResearch 490K review benchmark](https://research.aimultiple.com/open-source-embedding-models/)

**Verdict**: BGE-M3 confirmed. Best open-source embedding for RAG at practical size.

---

## 2. Vector Database

**Original recommendation**: LanceDB
**Updated to**: LanceDB (confirmed, no change)

**Validation**: LanceDB v0.29.2 released Feb 9, 2026. Jan 2026 newsletter shows
active development: HNSW-accelerated indexing, DuckDB integration, Uber-scale
storage. Contributors from Uber, Netflix, HuggingFace, Bytedance, Alibaba.

**Key advantage confirmed**: Zero-copy incremental updates via Lance columnar format.
This is critical for daily reddit-stash syncs. ChromaDB doesn't have this.

**LanceDB hybrid search confirmed** to work with BGE-M3 dense+sparse vectors.
Example: `table.search("query", query_type="hybrid")`.

**Sources**: [LanceDB Jan 2026 newsletter](https://lancedb.com/blog/newsletter-january-2026/), [PyPI v0.29.2](https://pypi.org/project/lancedb/), [LanceDB hybrid search docs](https://docs.lancedb.com/search/hybrid-search)

**Verdict**: LanceDB confirmed. Actively maintained, right features for our use case.

---

## 3. Graph Database

**Original recommendation**: KuzuDB
**Updated to**: NetworkX + SQLite (KuzuDB is dead)

**What changed**: KuzuDB was acquired by Apple in October 2025. Entire GitHub org
archived on Oct 10, 2025. Community discussion on HN explored alternatives:
- FalkorDBLite: New embedded mode, but spawns subprocess (not truly embedded)
- DuckDB + DuckPGQ: SQL/PGQ standard, but community extension still maturing
- NetworkX: Battle-tested since 2004, 14K stars, perfect for 1K-50K nodes

For personal knowledge graphs at our scale, a dedicated graph DB is overkill.
NetworkX handles all algorithms in-memory, SQLite persists the structure.

**Sources**: [The Register](https://www.theregister.com/2025/10/14/kuzudb_abandoned/), [Apple acquires Kuzu](https://macdailynews.com/2026/02/12/apple-acquires-graph-database-maker-kuzu/), [HN discussion](https://news.ycombinator.com/item?id=45560036)

**Verdict**: NetworkX + SQLite. Zero infrastructure risk, proven at scale.

---

## 4. Topic Modeling

**Original recommendation**: BERTopic
**Updated to**: BERTopic (confirmed, no change)

**Validation**: BERTopic remains the best for short social media text:
- 99.5% coherence on Reddit data (arXiv:2412.14486, tested on 12 subreddits)
- Compared against LDA (73.3%), NMF (86.6%), Top2Vec
- Supports zero-shot topic assignment (pre-define categories)
- FASTopic and QuaIIT are newer but not better for short text

**Sources**: [arXiv:2412.14486](https://arxiv.org/html/2412.14486v1), [BERTopic docs](https://maartengr.github.io/BERTopic/), [KDnuggets comparison](https://www.kdnuggets.com/2023/01/topic-modeling-approaches-top2vec-bertopic.html)

**Verdict**: BERTopic confirmed. No serious challenger for Reddit-length content.

---

## 5. RAG Framework

**Original recommendation**: LlamaIndex
**Updated to**: LlamaIndex (confirmed, strengthened)

**Validation**: Multiple 2026 comparisons confirm LlamaIndex is superior for
pure document RAG:
- IBM: "LlamaIndex is ideal for straightforward RAG"
- Leon Consulting: "LlamaIndex is objectively superior for RAG in 2026"
- Statsig: "LlamaIndex outperforms in data retrieval and indexing"

LangChain is better for complex multi-step agent workflows (500+ integrations).
For our use case (document retrieval + Q&A), LlamaIndex wins.

**Sources**: [IBM comparison](https://www.ibm.com/think/topics/llamaindex-vs-langchain), [Leon Consulting 2026](https://leonstaff.com/blogs/langchain-vs-llamaindex-rag-wars/), [AIMResearch](https://research.aimultiple.com/rag-frameworks/)

**Verdict**: LlamaIndex confirmed for RAG. Could add LangChain later for agent workflows.

---

## 6. Local LLM

**Original recommendation**: Ollama with Llama 3.1 8B ("GPT-3.5 quality")
**Updated to**: llama-cpp-python (primary) + Ollama (convenience), with Phi-4/Qwen 2.5

**What changed (MAJOR)**:
1. "GPT-3.5 quality" was wrong — current small models are GPT-4-level on many tasks
2. Added llama-cpp-python as primary option (zero overhead, direct C++ bindings)
3. Updated model recommendations to Feb 2026 landscape

**Current best small models**:
- Phi-4 (14B): Best quality/size ratio, matches 70B on reasoning
- Qwen 2.5 (7B): Best all-rounder, strong multilingual
- SmolLM3 (3B): Best tiny model, outperforms Llama-3.2-3B
- Gemma-3n-E2B (2B active): Google's on-device model

**llama.cpp vs Ollama**:
- llama.cpp: Zero overhead, 10-15% faster, full context window control, supports embeddings
- Ollama: Better UX (`ollama pull`), but runs a server with HTTP overhead
- Both use same underlying engine (Ollama wraps llama.cpp)

**Sources**: [BentoML SLMs 2026](https://www.bentoml.com/blog/the-best-open-source-small-language-models), [DataCamp top 15](https://www.datacamp.com/blog/top-small-language-models), [Openxcell llama.cpp vs Ollama](https://www.openxcell.com/blog/llama-cpp-vs-ollama/), [DecodesFuture benchmark](https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide)

**Verdict**: Support both. llama-cpp-python for power users, Ollama for convenience.

---

## 7. CLI Framework

**Original recommendation**: Click
**Updated to**: Typer

**What changed**: Typer is built on top of Click but uses Python type hints for
automatic CLI generation. Less boilerplate, automatic help docs, modern Python idiom.
Created by tiangolo (FastAPI author). Multiple 2026 guides recommend Typer over
raw Click for new projects.

**Sources**: [Typer docs](https://typer.tiangolo.com/alternatives/), [DevToolbox 2026 guide](https://devtoolbox.dedyn.io/blog/python-click-typer-cli-guide), [CodeCut comparison](https://codecut.ai/comparing-python-command-line-interface-tools-argparse-click-and-typer/)

**Verdict**: Typer. Modern, type-safe, less boilerplate than Click.

---

## 8. MCP Server SDK

**Original recommendation**: mcp Python SDK
**Updated to**: mcp Python SDK v1.26+ (confirmed, version updated)

**Validation**: Latest release v1.26.0 (Jan 24, 2026). FastMCP incorporated into
official SDK. Supports Python 3.10-3.13. Works with Claude Desktop, Cursor,
OpenAI Agents SDK. Best practices published by Docker, Snyk, and Phil Schmid.

**Sources**: [GitHub releases](https://github.com/modelcontextprotocol/python-sdk/releases), [PyPI](https://pypi.org/project/mcp/), [Docker best practices](https://www.docker.com/blog/mcp-server-best-practices/), [Phil Schmid](https://www.philschmid.de/mcp-best-practices)

**Verdict**: mcp SDK v1.26+ confirmed. Only option for official MCP servers.

---

## 9. Dashboard

**Original recommendation**: Streamlit
**Updated to**: Streamlit (confirmed, no change)

**Validation**: Streamlit remains the best for Python data app dashboards:
- Gradio: Better for ML model demos, not general dashboards
- Panel: More flexible but steeper learning curve
- Dash: Better for production but more boilerplate
- Streamlit: Fastest path from Python to web app, built-in chat UI

**Sources**: [MyScale comparison](https://www.myscale.com/blog/streamlit-vs-gradio-ultimate-showdown-python-dashboards/), [Anvil comparison](https://medium.com/anvil-works/streamlit-vs-gradio-vs-dash-vs-panel-vs-anvil-c2f86ad95ff3)

**Verdict**: Streamlit confirmed. Best for our use case (dashboard + chat).

---

## 10. Graph Visualization

**Original recommendation**: Pyvis
**Updated to**: Pyvis (confirmed, with Gravis as alternative)

**Validation**: Pyvis remains the go-to for NetworkX → interactive HTML:
- Direct NetworkX integration: `net.from_nx(G)`
- Interactive: drag, zoom, hover tooltips, physics simulation
- Alternative: Gravis (newer, different sidebar UI, comparable features)
- For large graphs (>10K nodes): consider graph-tool (C++ backend)

**Sources**: [Tom Sawyer comparison](https://blog.tomsawyer.com/python-graph-visualization-libraries), [Towards Data Science Gravis](https://towardsdatascience.com/the-new-best-python-package-for-visualising-network-graphs-e220d59e054e/)

**Verdict**: Pyvis confirmed. Proven, good NetworkX integration.

---

## Summary of Changes from Initial Research

| Component | Initial Choice | Final Choice | Changed? |
|-----------|---------------|-------------|----------|
| Embeddings | BGE-base-en-v1.5 | **BGE-M3** | Yes |
| Vector DB | LanceDB | LanceDB | No |
| Graph DB | KuzuDB | **NetworkX + SQLite** | Yes (Kuzu dead) |
| Topic Model | BERTopic | BERTopic | No |
| RAG Framework | LlamaIndex | LlamaIndex | No |
| Local LLM | Ollama only | **llama-cpp-python + Ollama** | Yes |
| LLM Models | Llama 3.1 8B | **Phi-4 / Qwen 2.5** | Yes |
| CLI | Click | **Typer** | Yes |
| MCP SDK | mcp SDK | mcp SDK v1.26+ | Updated version |
| Dashboard | Streamlit | Streamlit | No |
| Graph Viz | Pyvis | Pyvis | No |
