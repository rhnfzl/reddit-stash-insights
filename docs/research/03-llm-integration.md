# LLM Integration Research

## Problem Statement

Users want to "chat with their saved Reddit posts" — ask questions, get summaries,
and gain insights from their Reddit history using natural language.

## Architecture: RAG (Retrieval-Augmented Generation)

```
User query: "What did I save about Rust async?"
    |
    v
[1. Embed query] --> BGE-M3 (dense + sparse vectors)
    |
    v
[2. Hybrid search] --> LanceDB (BM25 + semantic + sparse, top 5-10 posts)
    |
    v
[3. Prompt LLM]   --> "Given these posts, answer: {query}"
    |
    v
[4. Response]     --> Grounded answer with source citations
```

## RAG Framework

### Recommended: LlamaIndex

- **Built for document RAG** (LangChain is for multi-step agents)
- **Native markdown support**: `SimpleDirectoryReader` auto-parses frontmatter
- **LanceDB integration**: First-class vector store adapter
- **3-line setup** for basic use:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./saved_posts").load_data()
index = VectorStoreIndex.from_documents(documents)
response = index.as_query_engine().query("Best Rust resources?")
```

### Why NOT LangChain

- LangChain is designed for complex agent workflows (multi-step reasoning)
- For document Q&A (our use case), LlamaIndex is simpler and more focused
- LangChain's API churn has been a pain point in 2025-2026

### Why NOT raw vector search (no framework)

If you only want search results (not LLM-generated answers), skip the framework.
Direct LanceDB queries are simpler and have zero LLM cost.
The framework only adds value when you want conversational Q&A.

## LLM Provider Abstraction

### Design: Pluggable Backend

Users choose via config:

```ini
[llm]
provider = llama-cpp       # or: ollama, openai, anthropic, google
model = phi-4-q4_k_m.gguf # or: qwen2.5:7b, gpt-4o-mini, claude-haiku-4-5, gemini-1.5-flash
```

### Local LLMs (via llama-cpp-python or Ollama)

| Model | Params | RAM | Best For | Notes |
|-------|--------|-----|----------|-------|
| **Phi-4** | 14B | 10GB | Best quality/size ratio | Matches many 70B models on reasoning |
| **Qwen 2.5** | 7B | 8GB | General RAG + multilingual | Strong all-rounder |
| **SmolLM3** | 3B | 4GB | Minimal hardware | Outperforms Llama-3.2-3B at same size |
| **Gemma-3n-E2B** | 5B (2B active) | 4GB | On-device / low resource | Selective parameter activation |
| **Qwen 2.5 Coder** | 7B | 8GB | Code-heavy subreddits | Beats GPT-4o on coding benchmarks |

> Note: The earlier recommendation of "Llama 3.1 8B = GPT-3.5 quality" was outdated.
> Current small models (Phi-4, Qwen 2.5) compete with GPT-4-level quality on many tasks.

**Two runtime options**:
- **llama-cpp-python**: Direct C++ bindings, zero overhead, pip-only install, supports embeddings AND inference
- **Ollama**: Simpler UX (`ollama pull`), 10-15% HTTP overhead, better for beginners

**Minimum hardware (CPU-only, no GPU):**
- 4GB RAM for 3B models, 8GB for 7B, 10-16GB for 14B
- 4+ core modern CPU (Intel i5+, M1+)
- ~5-10GB disk per model

**Apple Silicon performance:**
- M1: ~20-30 t/s (7B)
- M3: ~30-50 t/s (7B)

### Cloud APIs

| Provider | Model | Input/1M | Output/1M | Monthly (100 queries) |
|----------|-------|----------|-----------|----------------------|
| Google | Gemini 1.5 Flash | $0.08 | $0.30 | ~$0.87 |
| OpenAI | GPT-4o mini | $0.15 | $0.60 | ~$1.73 |
| Anthropic | Claude Haiku 4.5 | $1.00 | $5.00 | ~$6.00 |
| Anthropic | Claude Sonnet 4.5 | $3.00 | $15.00 | ~$18.00 |

**Free tier**: Google Gemini 1.5 Flash (1,500 queries/day free)

### Privacy Considerations

- **Local (Ollama)**: 100% private, no data leaves your machine
- **Cloud APIs**: Your Reddit data is sent to the provider
  - OpenAI: 30-day data retention
  - Anthropic: 0-day retention if opted out
  - Google: Data used for model improvement unless opted out
- **Recommendation**: Default to local; cloud as opt-in

## Intelligent Features Beyond Search

### 1. Time-Based Summaries
"Summarize what I saved this month about Rust"
- Filter by YAML `timestamp` → group by subreddit → LLM summarize

### 2. Interest Profiling
"What are my main interests?"
- TF-IDF on all post titles → weight by recency → top topics
- Output: "40% AI/ML, 30% Python, 20% Productivity, 10% Other"

### 3. Trend Detection
"How have my interests changed?"
- Monthly topic counts → percent change
- Output: "Your interest in ML papers increased 300% in Jan 2026"

### 4. Cross-Subreddit Insights
"Where do topics overlap?"
- Graph community detection on subreddit co-occurrence
- Output: "Posts about 'productivity' appear in r/productivity, r/gtd, r/ADHD"

### 5. Weekly Digest
- Cron job: summarize last 7 days of saves
- Output: markdown report or email

## Sources

- LlamaIndex vs LangChain 2026: https://rahulkolekar.medium.com
- Ollama benchmarks: https://localaimaster.com
- LLM API pricing Feb 2026: various provider pages
- RAG patterns: https://www.analyticsvidhya.com/blog/2024/01/rag-streamlit-chatbot/
