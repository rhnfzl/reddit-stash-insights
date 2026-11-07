# Retrieval evidence for the next release

Checked 2026-11-07. Agent reference notes for the release plan, not a claim of measured package improvements.

Users need to find the right saved post and read the passage that answers their question. The next release should make that behavior measurable before adding models.

## Repository observations

- `src/rsi/indexer/vector_store.py` already combines BM25 and dense retrieval with reciprocal rank fusion. Hybrid retrieval is not a missing feature.
- `SearchEngine.index_texts()` batches records to bound memory. Its `chunk_size` does not split a long post into searchable passages.
- `src/rsi/config.py` defaults `max_text_chars` to 32,000 characters. Passage indexing must address this separate input cap.
- `src/rsi/chat/prompt.py:build_context_block()` takes the first 500 characters of every retrieved result.
- `src/rsi/chat/engine.py` does not use the provider's existing `context_window` property when building messages.
- Existing local and OpenAI-compatible embedding backends allow controlled experiments without replacing LanceDB or adding a retrieval framework.

## Release recommendations

1. Freeze at least 50 human-reviewed, answerable questions with source IDs, evidence spans, expected answer facts, and relevance grades. Include exact lookup, paraphrase, comment-only evidence, long-tail evidence, and multi-source questions. Add a separate no-answer set. Generated drafts need review; model output alone is not ground truth.
2. Publish a small redistributable corpus and its labels with the runner. Keep the real homelab corpus and personal query text private. A public synthetic fixture can verify the runner but does not prove real archive quality.
3. Report recall@10, MRR with an explicit evaluation depth, and nDCG@10 separately. Also report failures, per-query results, corpus hash, code revision, model revision, latency, and judgment coverage. Never turn exceptions into zero-valued successful runs.
4. Preserve the existing BM25+dense fusion baseline. Split long post/comment bodies at paragraph boundaries with a token limit. Keep title, subreddit, parent/source IDs, source path, permalink, and offsets on each passage. Do not assume the archive contains every comment or a complete reply tree.
5. Retrieve passages but score source documents once. Deduplicate by stable source ID before computing document-level metrics. Retain passage evidence for answer checking. This prevents several chunks from one post inflating the score.
6. Replace the fixed 500-character truncation with bounded evidence selection. Reserve tokens for system instructions, query, history, and output. Count with the active model tokenizer where available. Select complete passages, trim history first, and disclose omitted evidence. Do not replace one silent cutoff with unlimited context.
7. Run controlled comparisons in order: current baseline; passage indexing; context-budget fix for answer quality; optional reranking; one compact embedding candidate. The context fix does not itself improve retrieval metrics. Keep it separate in reporting.
8. Keep new models optional until local results justify them. Graph extraction, fine-tuning, agentic query expansion, OCR, and multimodal retrieval add no demonstrated value for the current Markdown search problem.

These are project-specific design recommendations. The sources below motivate experiments; they do not establish gains on this archive.

## Evaluation contract

The maintained `ir_measures` documentation defines recall, reciprocal rank, and nDCG, with aliases `Recall -> R` and `MRR -> RR`. It supports aggregate and per-query output. Use it as an offline oracle for small hand-calculated metric tests, or as an optional eval dependency if that is simpler than maintaining formulas. See [measure definitions](https://ir-measur.es/en/latest/measures.html) and [Python examples](https://ir-measur.es/en/latest/getting-started.html). These are living documentation pages, read 2026-11-07.

Recall@10 uses all known relevant source documents as its denominator. Labeling only one convenient expected post understates that denominator. Pool results from lexical, dense, and hybrid baselines for review. Report recall against judged relevant documents and the limits of incomplete judgments. With one relevant item per question, recall@10 becomes success@10; say so.

Use grades 0 (irrelevant), 1 (useful context), and 2 (direct answer evidence). Freeze the binary relevance threshold and nDCG gain formulation. If the runner retrieves only 10 items, label reciprocal rank as MRR@10. To report untruncated MRR, search and disclose the full evaluation depth.

Reserve a fixed held-out subset before tuning. Fifty questions are an initial regression suite, not broad proof of search quality. Compare identical queries and corpus revisions. Include paired query deltas and bootstrap intervals. A wide interval means uncertainty, not a win. Separate retrieval relevance from generated-answer correctness and unsupported claims.

## Model and method shortlist

| Candidate | Verified evidence | Suggested use | Cost and constraint |
|---|---|---|---|
| Existing hybrid search | Already implemented in this repository | Default baseline | No new model or service |
| Paragraph/comment passages with inherited metadata | Context can be lost when documents are split; Anthropic describes contextualized chunks and BM25 | Required next-release experiment; use known metadata first | More rows and index work; no generated summaries required |
| Qwen3-Reranker-0.6B | Qwen release dated 2025-06-05; 0.6B parameters, 32K context, Apache-2.0 | Optional local rerank of a bounded candidate pool | Query-time inference for each pair; measure warm/cold latency and RAM |
| IBM Granite Embedding 311M Multilingual R2 | Model card release date 2026-04-29; 768 dimensions, 32,768 tokens, Apache-2.0 | First compact embedding experiment after passage baseline | New model needs a separate full index; local hardware compatibility untested |
| Microsoft Harrier OSS v1 270M/0.6B | Current official card: 640/1,024 dimensions, 32,768 tokens, MIT | Alternative compact experiment if Granite is unsuitable | Exact release date not established in extracted card; task prompt and pooling matter |
| Jina reranker v3.5 | Official announcement dated 2026-08-03; 0.6B listwise reranker | Watch list only | Non-commercial weights; listwise pool length/count limits and order sensitivity need checks |

Qwen's [release announcement](https://qwenlm.github.io/blog/qwen3-embedding/) confirms the date and Apache-2.0 license. Its [reranker card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) describes input formatting, instructions, and the model's evaluation. Those benchmark scores use a particular candidate pool and cannot be copied into this package's claims.

The [Granite model card](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2) states the release date, architecture, license, dimensions, context limit, and language limits. Its [raw card](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/raw/main/README.md) includes the explicit release date. The [Harrier model card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b) and [raw card](https://huggingface.co/microsoft/harrier-oss-v1-0.6b/raw/main/README.md) provide the family table, MIT license, last-token pooling, normalization, and retrieval query prompt. Neither model is verified against the project's installed runtime.

Jina's [2026-08-03 announcement](https://jina.ai/news/jina-reranker-v3-5-faster-listwise-reranking-hybrid-attention-self-distillation) reports efficiency gains on its own benchmarks and explicitly describes remaining input and domain limits. The [official license FAQ](https://jina.ai/reranker) states CC-BY-NC-4.0 for current reranker weights. This makes it unsuitable as an unrestricted default for an MIT package's users without separate license review.

[Sentence Transformers' retrieve-and-rerank guide](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) describes the two-stage design: retrieve a candidate pool, then score query-document pairs. Reranking cannot recover a document absent from that pool. Start with a fixed pool such as 30 candidates as a test setting, not a claimed optimum. Increase only when candidate recall and hardware measurements justify it.

Anthropic's [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) article (2024-09-19) studies chunk context, BM25, and reranking. Its numerical gains are from its experiments. Copy the experiment structure, not the gains. Start with deterministic Reddit metadata rather than an extra LLM indexing pass.

[Microsoft's GraphRAG documentation](https://microsoft.github.io/graphrag/) describes entity extraction, community hierarchy, and summaries for local/entity and global/corpus questions. That is a different capability from recovering a saved post. Defer it until real user questions and labeled failures require it.

## Completion criteria

- The public fixture reproduces results without personal data, credentials, or a model download for metric-unit tests.
- Every question has reviewed source evidence; no-answer questions are reported separately.
- Duplicate passages, missing IDs, malformed grades, empty runs, and backend failures have explicit checks.
- Before/after runs use the same frozen corpus and labels. Existing scores and exceptions are visible per query.
- Long-post tests retrieve evidence after character 500 and after the old 32,000-character indexing cap.
- Source citations resolve to the retrieved evidence across CLI, UI, and MCP.
- Chat requests stay inside the selected provider's actual configured context budget.
- A candidate model ships as default only after held-out gains justify measured latency, memory, and indexing cost. Keep the old index until the new index passes checks.

## Research limits

Tavily Research with `model=pro` completed the broad review. Its generated advice incorrectly treated the existing 500-character cutoff as a requirement and included secondary sources. Those recommendations were rejected. Targeted advanced Tavily Search and Extract verified the primary pages cited above. No homelab text was uploaded or sent to a hosted model. The search covers practical candidates available by 2026-11-07; it does not prove any model is the latest or best globally. No local ML benchmark, download, or dependency-runtime claim was made.
