"""Measure search quality against frozen source-thread relevance labels."""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
from importlib import metadata, resources
from pathlib import Path
from typing import Any, Callable, Iterable

from rsi.core.identity import canonical_doc_id

try:
    import resource
except ImportError:  # Windows does not expose process peak RSS in the standard library.
    resource = None  # type: ignore[assignment]

_DOC_ID = re.compile(r"^(post|comment):[a-zA-Z0-9_-]+$")


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_DOC_ID.fullmatch(value))


def score_ranking(ranked_ids: Iterable[str], qrels: dict[str, int], k: int = 10) -> dict[str, float]:
    """Score unique source threads; unknown labels contribute no judged gain."""
    if type(k) is not int or k < 1:
        raise ValueError("cutoff must be a positive integer")
    if any(not _valid_id(doc) or type(grade) is not int or grade not in (0, 1, 2)
           for doc, grade in qrels.items()):
        raise ValueError("qrels require canonical document IDs and integer grades 0, 1, 2")
    relevant = {doc for doc, grade in qrels.items() if grade > 0}
    if not relevant:
        raise ValueError("answerable query has no judged relevant documents")
    values = list(ranked_ids)
    if any(not _valid_id(doc) for doc in values):
        raise ValueError("rankings require canonical document IDs")
    ranked = list(dict.fromkeys(values))[:k]
    gains = [2 ** qrels.get(doc, 0) - 1 for doc in ranked]
    dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
    ideal = sum((2 ** grade - 1) / math.log2(rank + 2)
                for rank, grade in enumerate(sorted(qrels.values(), reverse=True)[:k]))
    first = next((rank for rank, doc in enumerate(ranked, 1) if doc in relevant), None)
    return {
        f"recall@{k}": len(set(ranked) & relevant) / len(relevant),
        f"mrr@{k}": 1 / first if first else 0.0,
        f"ndcg@{k}": dcg / ideal,
    }


def retrieve_threads(
    search: Callable[..., list[dict[str, Any]]], query: str, mode: str, subreddit: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Keep first occurrences and increase the bounded pool when chunks crowd it."""
    unique: dict[str, dict[str, Any]] = {}
    for depth in (50, 100, 200):
        rows = search(query, limit=depth, mode=mode, subreddit=subreddit)
        if not isinstance(rows, list) or len(rows) > depth:
            raise ValueError("retriever returned an invalid candidate ranking")
        unique = {}
        for row in rows:
            unique.setdefault(canonical_doc_id(row), row)
        if len(unique) >= 10 or len(rows) < depth:
            break
    return list(unique.values())[:10], depth


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def snapshot_manifest(corpus: Path) -> dict[str, Any]:
    """Hash raw Markdown bytes and relative paths without leaking absolute paths."""
    files = {path.relative_to(corpus).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(corpus.rglob("*.md"))}
    if not files:
        raise ValueError("snapshot has no Markdown files")
    return {"files": files, "sha256": _digest(files)}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}:{number}: expected a JSON object")
            rows.append(row)
    return rows


def load_dataset(
    dataset: Path, qrels: Path, corpus: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], dict[str, Any]]:
    """Validate the frozen corpus, questions and labels before searching."""
    from rsi.core.scanner import scan_directory

    expected = json.loads(dataset.with_name("snapshot.json").read_text(encoding="utf-8"))
    if not isinstance(expected, dict) or not isinstance(expected.get("files"), dict):
        raise ValueError("snapshot.json must be an object with a files mapping and sha256")
    if "documents" in expected and (not isinstance(expected["documents"], list)
                                     or any(not _valid_id(doc) for doc in expected["documents"])):
        raise ValueError("snapshot document IDs must be canonical strings")
    snapshot = snapshot_manifest(corpus)
    if expected.get("files") != snapshot["files"] or expected.get("sha256") != snapshot["sha256"]:
        raise ValueError("corpus snapshot hash does not match snapshot.json")
    scan = scan_directory(corpus)
    if scan.errors:
        raise ValueError(f"snapshot contains {len(scan.errors)} parser errors")
    documents = {
        canonical_doc_id({"id": item.id, "content_type": kind, "permalink": item.permalink})
        for kind, items in (("post", scan.posts), ("comment", scan.comments)) for item in items
    }
    if "documents" in expected and set(expected["documents"]) != documents:
        raise ValueError("snapshot document IDs do not match parsed corpus")
    evidence_text: dict[str, str] = {}
    for kind, items in (("post", scan.posts), ("comment", scan.comments)):
        for item in items:
            source_doc = canonical_doc_id({"id": item.id, "content_type": kind, "permalink": item.permalink})
            evidence_text[source_doc] = evidence_text.get(source_doc, "") + "\n" + item.search_text()
    queries = _jsonl(dataset)
    seen = set()
    labels: dict[str, dict[str, int]] = {}
    for row in queries:
        qid = row.get("query_id")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError(f"missing or duplicate query ID: {qid!r}")
        seen.add(qid)
        if (not isinstance(row.get("query"), str) or not row["query"].strip()
                or row.get("split") not in ("dev", "test") or type(row.get("answerable")) is not bool
                or not isinstance(row.get("tags"), list) or not isinstance(row.get("evidence"), list)
                or not isinstance(row.get("reference_answer"), str)):
            raise ValueError(f"invalid question fields for {qid}")
        evidence = row["evidence"]
        evidence_ids = [e.get("doc_id") if isinstance(e, dict) else e for e in evidence]
        if any(not _valid_id(doc) or doc not in documents for doc in evidence_ids):
            raise ValueError(f"evidence references a missing document for {qid}")
        for entry in evidence:
            if isinstance(entry, dict) and "quote" in entry:
                if not isinstance(entry["quote"], str) or not entry["quote"] or entry["quote"] not in evidence_text[entry["doc_id"]]:
                    raise ValueError(f"evidence quote does not occur in frozen source for {qid}")
        if row["answerable"] and (not evidence or not row["reference_answer"].strip()):
            raise ValueError(f"answerable query requires reference answer and evidence: {qid}")
        if not row["answerable"] and evidence:
            raise ValueError(f"absent query has evidence: {qid}")
        labels[qid] = {}
    if not queries:
        raise ValueError("dataset has no questions")
    for row in _jsonl(qrels):
        qid, doc, grade = row.get("query_id"), row.get("doc_id"), row.get("grade")
        if not isinstance(qid, str) or qid not in labels or not _valid_id(doc) or doc not in documents:
            raise ValueError("qrel references a missing query or document")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError("qrel grade must be an integer 0, 1 or 2")
        if doc in labels[qid]:
            raise ValueError(f"duplicate qrel for {qid}/{doc}")
        labels[qid][doc] = grade
    for row in queries:
        relevant = any(grade > 0 for grade in labels[row["query_id"]].values())
        if row["answerable"] != relevant:
            raise ValueError(f"answerability disagrees with qrels for {row['query_id']}")
    snapshot["documents"] = sorted(documents)
    snapshot["aliases"] = {f"{kind}:{item.id}": canonical_doc_id(
        {"id": item.id, "content_type": kind, "permalink": item.permalink})
        for kind, items in (("post", scan.posts), ("comment", scan.comments)) for item in items}
    return queries, labels, snapshot


def index_fingerprint(store: Any, snapshot: dict[str, Any]) -> dict[str, Any]:
    """Reject absent or unrelated indexes and identify the stored retrieval texts."""
    if not store.has_table() or store.count() == 0:
        raise ValueError("evaluation index has no searchable source records")
    # Legacy tables lack source permalink columns; the frozen scanner supplies their aliases.
    columns = set(store._table.schema.names)
    if not {"id", "content_type", "text"}.issubset(columns):
        raise ValueError("evaluation index lacks source ID, type or text columns")
    selected = sorted(columns & {"id", "content_type", "text", "permalink", "parent_doc_id"})
    rows = store._table.search().select(selected).limit(store.count()).to_list()
    fingerprints = []
    documents = set()
    aliases = snapshot.get("aliases", {})
    for row in rows:
        doc = canonical_doc_id(row)
        doc = aliases.get(doc, doc)
        documents.add(doc)
        if not isinstance(row["text"], str):
            raise ValueError("evaluation index contains a non-text source record")
        fingerprints.append({"id": row["id"], "content_type": row["content_type"], "doc_id": doc,
                             "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest()})
    expected = set(snapshot["documents"])
    if documents != expected:
        raise ValueError(f"index sources do not match frozen snapshot: {len(expected - documents)} missing, "
                         f"{len(documents - expected)} unexpected")
    fingerprints.sort(key=lambda row: (row["content_type"], row["id"]))
    return {"source_threads": len(documents), "stored_texts_sha256": _digest(fingerprints)}


def export_fixture(destination: Path) -> None:
    """Copy packaged resources, including from an installed wheel, to a directory."""
    def copy(source: Any, target: Path) -> None:
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            for child in sorted(source.iterdir(), key=lambda item: item.name):
                if child.name != "__pycache__":
                    copy(child, target / child.name)
        else:
            if target.exists():
                raise ValueError(f"refusing to overwrite fixture file: {target}")
            target.write_bytes(source.read_bytes())
    copy(resources.files("rsi").joinpath("eval_fixture"), destination)


def _code_identity() -> dict[str, Any]:
    from rsi import __version__

    root = Path(__file__).resolve().parents[2]
    revision = None
    if (root / ".git").exists():
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        if result.returncode == 0:
            revision = result.stdout.strip()
    source_files = {path.relative_to(Path(__file__).parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(Path(__file__).parent.rglob("*.py"))}
    dependencies = {}
    for name in ("lancedb", "FlagEmbedding", "sentence-transformers", "openai"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"version": __version__, "revision": revision, "python_sources_sha256": _digest(source_files),
            "dependencies": dependencies}


def evaluate(
    queries: list[dict[str, Any]], labels: dict[str, dict[str, int]], snapshot: dict[str, Any],
    search: Callable[..., list[dict[str, Any]]], *, mode: str, split: str, identity: dict[str, Any],
) -> dict[str, Any]:
    """Keep per-query failures visible and exclude absent cases from retrieval scores."""
    if mode not in ("keyword", "semantic", "hybrid") or split not in ("dev", "test", "all"):
        raise ValueError("invalid search mode or split")
    selected = sorted((row for row in queries if split == "all" or row["split"] == split),
                      key=lambda row: row["query_id"])
    if not selected:
        raise ValueError("selected split has no questions")
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for query in selected:
        row: dict[str, Any] = {"query_id": query["query_id"], "query": query["query"], "split": query["split"],
                               "tags": query["tags"], "answerable": query["answerable"], "metrics": None,
                               "ranking": [], "error": None}
        begin = time.perf_counter()
        try:
            def mapped_search(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
                rows = search(*args, **kwargs)
                aliases = snapshot.get("aliases", {})
                return [{**item, "parent_doc_id": aliases.get(canonical_doc_id(item), canonical_doc_id(item))}
                        for item in rows]

            ranking, depth = retrieve_threads(mapped_search, query["query"], mode, query.get("subreddit"))
            ids = [canonical_doc_id(item) for item in ranking]
            if any(doc not in snapshot["documents"] for doc in ids):
                raise ValueError("ranking references a document outside the frozen snapshot")
            row.update({"ranking": [{"rank": rank, "doc_id": doc, "raw_id": str(item.get("id", "")),
                                     "evidence_id": item.get("source_doc_id") or f"{item.get('content_type')}:{item.get('id')}",
                                     "grade": labels[query["query_id"]].get(doc)}
                                    for rank, (doc, item) in enumerate(zip(ids, ranking), 1)],
                        "candidate_depth": depth, "unique_result_count": len(ids),
                        "judged_count": sum(doc in labels[query["query_id"]] for doc in ids),
                        "retrieved_count": len(ids)})
            if query["answerable"]:
                row["metrics"] = score_ranking(ids, labels[query["query_id"]])
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["latency_ms"] = (time.perf_counter() - begin) * 1000
        results.append(row)
    answerable = [row for row in results if row["answerable"]]
    errors = sum(row["error"] is not None for row in results)
    metrics = None
    if answerable and not errors:
        metrics = {key: sum(row["metrics"][key] for row in answerable) / len(answerable)
                   for key in ("recall@10", "mrr@10", "ndcg@10")}
    retrieved = sum(row.get("retrieved_count", 0) for row in answerable)
    judged = sum(row.get("judged_count", 0) for row in answerable)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
    latencies = sorted(row["latency_ms"] for row in results)
    return {
        "schema": 1, "mode": mode, "split": split, "snapshot_sha256": snapshot["sha256"],
        "identity": {**_code_identity(), **identity}, "answerable_count": len(answerable),
        "absent_count": len(results) - len(answerable), "error_count": errors, "metrics": metrics,
        "judged_coverage": judged / retrieved if retrieved else None,
        "judged_coverage_counts": {"judged": judged, "retrieved": retrieved},
        "limitations": "Recall covers the judged relevant set. Unjudged results are unknown, not proven irrelevant. "
                       "Synthetic scores do not establish general retrieval superiority. Timings and RSS vary per run.",
        "absent_answers": {"evaluated": False, "abstention_count": None, "unsupported_answer_count": None,
                           "reason": "Retrieval only; no chat model or answer judge is invoked."},
        "latency_ms": (time.perf_counter() - started) * 1000,
        "latency_p50_ms": latencies[math.ceil(len(latencies) * 0.5) - 1],
        "latency_p95_ms": latencies[math.ceil(len(latencies) * 0.95) - 1],
        "latency_percentile_method": "nearest rank, per-query elapsed milliseconds",
        "peak_rss_bytes": None if peak is None else int(peak if sys.platform == "darwin" else peak * 1024),
        "peak_rss_scope": "process lifetime peak including setup; null when unsupported",
        "queries": results,
    }
