# Reproduce search evaluation

Search evaluation shows whether saved source threads appear near the top of search results.
It measures retrieval against frozen labels, without a chat model or an LLM judge.

## Run the public fixture

Install the index extra. It includes LanceDB, but does not install an embedding model:

```sh
python -m pip install 'reddit-stash-insights[index]'
rsi eval --fixture --mode keyword --split all --output /tmp/rsi-public-evaluation.json
```

Before publication, install from the source checkout with `python -m pip install '.[index]'`.
The command copies packaged resources using `importlib.resources` and builds a temporary FTS index.
The temporary index has no vectors. It never loads an embedding model or calls a hosted provider.
It does not use the configured archive index. The report remains at the requested output path.

The package contains 50 original fictional fact questions and 10 absent-answer controls.
Answerable questions use 20 development and 30 test cases. Absent cases use four development and six test cases.
Source threads do not cross splits. Six questions use subreddit filters, four use languages other than English,
and ten use standalone comments. Ten facts appear after 32,000 characters, all in the test split.
Each answerable question has one direct source-thread label. All other fixture threads have explicit grade-zero labels.
The scorer's unit tests also cover partial grade-one labels.

All 50 answer/evidence pairs, 60 Markdown files, 2,500 qrels and snapshot hashes received independent agent review.
Human review remains pending. The long threads contain repeated filler and the short threads contain single facts.
This fixture checks installation and scoring behavior. It is not evidence of general retrieval superiority.

`src/rsi/eval_fixture/expected.json` freezes reference keyword rankings and scores with its LanceDB version.
Compare matching versions when checking ranking differences. Dependency changes can alter FTS ties.
Timing and memory fields vary between runs; rankings, labels and scores use deterministic ordering.

## Verify an installed wheel

Run these commands from the checkout after building the wheel:

```sh
python -m build
python -m venv /tmp/rsi-wheel-eval
/tmp/rsi-wheel-eval/bin/python -m pip install './dist/reddit_stash_insights-0.1.0-py3-none-any.whl[index]'
cd /tmp
/tmp/rsi-wheel-eval/bin/rsi eval --fixture --mode keyword --split all --output /tmp/rsi-wheel-report.json
```

This uses installed package resources and requires neither the source checkout nor a personal archive.
The fixture is covered by the package's existing `src/rsi` wheel target.

## Prepare a private evaluation

Keep private corpora, labels and reports outside the repository.
Freeze a copied archive and use an isolated evaluation index. Do not rebuild a live archive index.
Create `questions.jsonl`, `qrels.jsonl`, and `snapshot.json` next to each other.

Each question is one JSON object:

```json
{"query_id":"q1","query":"When does the Cedar backup start?","split":"dev","tags":["exact"],"answerable":true,"reference_answer":"03:40 UTC","evidence":[{"doc_id":"post:f001","evidence_id":"post:f001","quote":"The Cedar backup job starts at 03:40 UTC."}]}
```

Evidence can also be a list of canonical document ID strings.
Object evidence requires `doc_id`; optional `quote` must occur in that parsed source thread.
Keep a separate `evidence_id` when the source is a comment.
The optional `subreddit` question field applies the same filter as ordinary search.

Each qrel is one JSON object, with an integer grade of zero, one or two:

```json
{"query_id":"q1","doc_id":"post:f001","grade":2}
```

Use zero for irrelevant, one for partial evidence and two for direct evidence.
An answerable question needs a reference answer, evidence and at least one positive qrel.
Absent-answer questions use `answerable:false`, `evidence:[]` and no positive qrels.
Do not treat a fact absent from one thread as absent from the whole archive.

Create the snapshot from raw Markdown bytes and relative paths:

```python
import json
from pathlib import Path
from rsi.evaluation import snapshot_manifest

corpus = Path('/private/eval/corpus')
Path('/private/eval/snapshot.json').write_text(
    json.dumps(snapshot_manifest(corpus), indent=2, sort_keys=True) + '\n'
)
```

`files` maps each relative Markdown path to its SHA-256 hash.
`sha256` hashes that mapping's UTF-8 JSON with sorted keys and compact separators.
An optional `documents` list must match canonical IDs from the parsed, deduplicated corpus.
Corpus changes, duplicate query IDs or qrels, invalid grades and missing source references stop evaluation.
Empty indexes and indexes with different canonical source IDs also stop evaluation.
Reports fingerprint stored texts and source IDs, so compared indexes have an explicit content identity.

Source-thread IDs use `post:id`. Standalone comments resolve through their existing permalink.
Comments without a known thread retain `comment:id`. Chunk records use `parent_doc_id`.
The evaluator builds aliases from the same scanner-selected source files for legacy indexes without permalinks.
No external source lookup is used.

Run evaluation with every input path explicit:

```sh
RSI_EMBEDDING_PROVIDER=local RSI_EMBEDDING_MODEL=BAAI/bge-m3 \
  rsi eval --dataset /private/eval/questions.jsonl --qrels /private/eval/qrels.jsonl \
  --corpus /private/eval/corpus --db-path /private/eval/index \
  --mode hybrid --split dev --output /private/eval/dev-report.json
```

Keyword mode makes no embedding call. Semantic and hybrid modes use the configured embedding provider.
Index embedding identity and schema checks still apply. Provision the local model before an offline run.
A hosted embedding provider requires `--allow-remote`. Its destination appears before any query can be sent.
No chat provider is selected or invoked.

## Read the scores

The cutoff is ten unique source threads, ordered by their first retrieved occurrence.
The candidate pool starts at 50 and increases to 100 and 200 as needed.
The report records candidate depth and unique result count. Exhausted pools can return fewer than ten threads.

- Recall@10 divides retrieved judged-relevant threads by all judged-relevant threads. Grades one and two count.
- MRR@10 is the reciprocal rank of the first judged-relevant thread, or zero for a miss.
- nDCG@10 discounts gain by `log2(rank + 1)`, with gain `2^grade - 1`, then divides by ideal gain at ten.

Absent-answer cases do not enter retrieval metric denominators.
The report sets answer abstention and unsupported-answer counts to null because no answer model runs.
A retrieval error remains visible per query and invalidates aggregate scores. It never becomes a successful zero score.

Judged coverage reports judged retrieved threads divided by all retrieved threads for answerable questions.
Unknown labels stay null in rankings. Recall covers only the judged relevant set.
For private experiments, pool keyword, semantic and hybrid top-20 results plus known evidence before reviewing labels.
Record that incomplete pools can miss relevant threads.

Each report includes snapshot and label hashes, code source hash, package version, index metadata,
embedding identity, search configuration, ordered rankings, per-query scores and errors.
Source checkout runs also record the checkout revision. Installed wheels do not report an unrelated checkout revision.
Reports include per-query and total elapsed milliseconds, nearest-rank p50/p95 latency and process-lifetime peak RSS in bytes.
Peak RSS includes setup and is null on systems without the standard-library `resource` module.

Freeze development settings before opening test results. Compare paired scores on the same frozen corpus and splits.
Report exact deltas, affected queries and uncertainty. Fifty questions give directional evidence, not a general superiority claim.
