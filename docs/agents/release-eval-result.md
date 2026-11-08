# Reproducible evaluation result

Users can run a packaged synthetic retrieval check without their archive or an embedding model.
The `rsi eval` command also evaluates an explicit frozen corpus and index with reviewed relevance labels.

## Files

- `src/rsi/evaluation.py`: scoring, source-thread identity, frozen snapshot validation, index fingerprinting and JSON reports.
- `src/rsi/eval_fixture/`: 50 original factual questions, 10 absent controls, 60 source-compatible Markdown files, exhaustive qrels, hashes and reference keyword rankings.
- `src/rsi/cli.py`: the `eval` command only.
- `tests/test_evaluation.py`, `tests/test_cli_eval.py`: scoring, bounded deduplication, validation and local-only CLI regressions.
- `docs/evaluation.md`: installed-wheel reproduction, formats, commands, metric definitions and limits.
- `noxfile.py`, `.github/workflows/compatibility.yml`: new test modules in core and service/index-capable checks.

## Validation

Tests first failed for the missing evaluator and CLI command. Later regressions failed before their fixes.
The regressions cover empty or unrelated indexes, malformed snapshot/index JSON, unhashable IDs, missing evidence,
canonical comment mapping and terminal permalink IDs.

Project runtime: `/Users/rehan-8v/projects/personal/reddit-stash-insights/.venv/bin/python` (Python 3.14).

- `python -m unittest tests.test_evaluation tests.test_cli_eval -v`: 21 passing tests.
- `RSI_SKIP_SLOW_TESTS=1 python -m unittest discover -v`: 199 tests, 10 skipped, passing.
- `ruff check src/ tests/`: passing.
- `mypy src/rsi/`: passing across 28 source files.
- `rsi eval --fixture --split all`: 50 answerable, 10 absent, zero errors.
- Public FTS reference: recall@10 1.0, MRR@10 0.98, nDCG@10 0.9852371901428583.

The existing VectorStore FTS creation API emits a LanceDB deprecation warning. Its behavior passes the tests.

## Review and limits

The public fixture received independent agent review of all 50 source/evidence pairs and all source hashes.
Human review remains pending. The fixture contains single-fact short threads and repetitive long-thread padding.
All ten public tail facts are in the test split. Partial grade-one scoring has hand-calculated unit tests;
the public qrels contain grades zero and two.

No private held-out questions were read by this worker. Only frozen source files and the snapshot ID set
were inspected to diagnose an independent manifest mismatch. Root owns resolution and private baseline runs.

Reports keep unknown labels null, exclude absent cases from retrieval denominators, and leave answer-abstention metrics unevaluated.
A per-query error invalidates aggregate scores and exits nonzero. No chat model runs.
Hosted embeddings require explicit `--allow-remote` and show their destination before sending query text.
Keyword mode never creates an embedding model. Fixture mode never touches the configured archive index.

Root owns installed-wheel verification, package publication, full local-model baselines and retrieval changes.
No commits were created by this worker. No new dependencies were added.
