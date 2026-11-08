# Changelog

## 0.1.1

Compare search settings on private data without publishing personal results.

- Keep personal evaluation data and all derived reports local. Add repository privacy checks and document the public fictional fixture.
- Evaluate supplied indexes through the public source-thread search interface; keep raw passage diagnostics separate.
- Document local dataset templates, frozen test splits and the limits of retrieval-only scores.

## 0.1.0

Saved Reddit threads can be searched and cited without losing their later comments.
The first public release also includes a reproducible search evaluation and an installed web interface.

- Add `rsi eval` with recall@10, MRR@10 and nDCG@10, raw rankings and input fingerprints.
- Ship 50 fictional questions, reference answers and relevance labels, plus ten absent-answer controls.
- Split posts and comments into bounded passages with source offsets and comment references.
- Preserve full saved comment exports, including parent threads, replies and text by other authors. Reindex existing comment archives to include this context.
- Remove the indexing character cap and configure embedding input capacity explicitly.
- Group search results by source thread and preserve subreddit filters in every search mode.
- Fit chat evidence and history into provider token budgets, with citations for included passages.
- Require explicit cloud fallback permission and show the selected destination.
- Add `rsi ui`, an installed-wheel demo, GitHub Discussions and a failed-search issue form.
- Remove unsupported graph and analytics promises and unverified comparison claims.

Existing legacy indexes remain readable. Indexing requires a fresh directory or an explicit rebuild.
A rebuild discards the selected index table. Keep the source archive and back up the existing index first.
Comment source references follow exported nesting and fallback links; they do not reconstruct a complete comment graph.
The public fixture checks installation and scoring behavior. It does not establish general search quality.
