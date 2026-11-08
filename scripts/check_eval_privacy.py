"""Reject personal evaluation artifacts while allowing the fictional public fixture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Iterable


FIXTURE = "src/rsi/eval_fixture/"
PRIVATE_SUFFIXES = (
    "-plan-evidence-supported-answers-for-saved-reddit-posts.html",
    "-plan-chat-generation-and-fresh-evaluation-experiments.html",
    "-research-evidence-release-evaluation.html",
    "-research-chat-generation-fresh-evaluation.html",
    "-evidence-release-evaluation.json",
    "-chat-generation-fresh-evaluation.json",
)
LEGACY_PAGES = (
    "docs/human-html/2026-11-07-plan-reddit-stash-insights-next-release.html",
    "docs/human-html/2026-11-07-research-reddit-stash-insights-0-1-0-case-study.html",
)
PUBLIC_MARKER = '<meta name="evaluation-data" content="fictional-only">'
LEGACY_AGGREGATE = "docs/references/2026-11-08-rsi-0.1.0-aggregate.json"
WITHDRAWN_RECORD = {
    "schema": 1,
    "status": "withdrawn-personal-evaluation",
    "documentation": "../evaluation.md",
    "public_fixture_command": "rsi eval --fixture --mode keyword --split all",
}


def personal_record(value: object) -> bool:
    if isinstance(value, list):
        return any(personal_record(item) for item in value)
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("query_id"), str) and ("query" in value or "doc_id" in value):
        return True
    if any(isinstance(value.get(key), str) for key in ("raw_snapshot_sha256", "questions_sha256", "qrels_sha256")):
        return True
    if isinstance(value.get("snapshot_sha256"), str) and ("queries" in value or "identity" in value):
        return True
    if type(value.get("schema")) is int and {"paired_results", "aggregates", "observables"} & value.keys():
        return True
    return any(personal_record(item) for item in value.values())


def violation(name: str, raw: bytes) -> str | None:
    path = PurePosixPath(name)
    if name.startswith(FIXTURE):
        return None
    if name.startswith("docs/agents/"):
        return "private agent working notes"
    if {"data", "reddit", "eval-reports", ".personal-eval", "personal-eval"} & set(path.parts):
        return "personal evaluation directory"
    if any(part.endswith(".lance") for part in path.parts):
        return "archive index"
    if name.endswith(PRIVATE_SUFFIXES):
        return "personal study report"
    if path.suffix == ".jsonl" or path.name in {"questions.json", "qrels.json", "snapshot.json"}:
        return "evaluation dataset or snapshot outside the fictional fixture"
    if path.name.startswith(("POST_", "COMMENT_", "SAVED_", "UPVOTE_", "GDPR_")):
        return "archive content outside the fictional fixture"
    if name in LEGACY_PAGES and PUBLIC_MARKER.encode() not in raw:
        return "legacy page has not been replaced with fictional-only documentation"
    if path.suffix in {".html", ".md", ".json"} and any(s.encode() in raw for s in PRIVATE_SUFFIXES):
        return "reference to a personal study report"
    if path.suffix == ".json":
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return "JSON cannot be checked"
        if name == LEGACY_AGGREGATE and value != WITHDRAWN_RECORD:
            return "legacy aggregate must contain only the withdrawal marker"
        if personal_record(value):
            return "evaluation records, corpus fingerprints or derived results"
    return None


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def check(root: Path, ref: str | None) -> list[tuple[str, str]]:
    args = ("ls-tree", "-r", "--name-only", "-z", ref) if ref else (
        "ls-files", "--cached", "--others", "--exclude-standard", "-z",
    )
    names = sorted(set(git(root, *args).decode().rstrip("\0").split("\0")))
    failures = []
    for name in filter(None, names):
        path = root / name
        if ref is None and not path.is_file():
            continue
        raw = git(root, "show", f"{ref}:{name}") if ref else path.read_bytes()
        reason = violation(name, raw)
        if reason:
            failures.append((name, reason))
    return failures


def outgoing_refs(lines: Iterable[str]) -> list[str]:
    refs = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4 or len(parts[1]) not in (40, 64) or any(
            character not in "0123456789abcdefABCDEF" for character in parts[1]
        ):
            raise ValueError("invalid pre-push ref record")
        if set(parts[1]) != {"0"}:
            refs.append(parts[1])
    return refs


def self_check() -> None:
    assert violation(FIXTURE + "questions.jsonl", b'{"query_id":"fictional"}') is None
    assert violation(FIXTURE + "corpus/POST_fictional.md", b"fictional") is None
    assert violation("data/eval/report.json", b"{}")
    assert violation("docs/agents/release-progress.md", b"private working notes")
    assert violation("index/vectors.lance/data/example", b"binary")
    assert violation("questions.jsonl", b"{}")
    assert violation("archive/SAVED_example.md", b"fictional")
    assert violation("docs/run.json", b'{"queries":[{"query_id":"q","query":"fictional"}]}')
    assert violation("docs/run.json", b'{"corpus":{"raw_snapshot_sha256":"fictional"}}')
    assert violation("docs/run.json", b'{"schema":1,"aggregates":{}}')
    assert violation("docs/run.json", b"invalid JSON")
    assert violation(LEGACY_PAGES[0], b"old page")
    assert violation(LEGACY_PAGES[0], PUBLIC_MARKER.encode()) is None
    assert violation("docs/index.html", PRIVATE_SUFFIXES[0].encode())
    assert violation("docs/human-html/docs-index.json", b'{"version":1,"artifacts":[]}') is None
    assert violation("docs/reference.json", b'{"schema":1,"status":"withdrawn-personal-evaluation"}') is None
    assert violation("schemas/questions.schema.json", b'{"properties":{"query_id":{"type":"string"}}}') is None
    assert violation(LEGACY_AGGREGATE, json.dumps(WITHDRAWN_RECORD).encode()) is None
    assert violation(LEGACY_AGGREGATE, json.dumps({**WITHDRAWN_RECORD, "metrics": {}}).encode())
    assert outgoing_refs([f"refs/heads/example {'a' * 40} refs/heads/example {'b' * 40}"]) == ["a" * 40]
    assert outgoing_refs([f"(delete) {'0' * 40} refs/heads/example {'b' * 40}"]) == []
    try:
        outgoing_refs(["malformed"])
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed push input passed the privacy check")
    print("PASS: 22 fictional privacy checks")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--working-tree", action="store_true", help="Inspect proposed local changes without staging them")
    mode.add_argument("--ref", default=None, help="Inspect a committed tree; defaults to HEAD")
    mode.add_argument("--pre-push", action="store_true", help="Read outgoing Git hook refs from stdin")
    mode.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    refs: list[str | None] = [None if args.working_tree else args.ref or "HEAD"]
    if args.pre_push:
        try:
            refs = outgoing_refs(sys.stdin)
        except ValueError as error:
            parser.error(str(error))
    failures = [(ref, name, reason) for ref in refs for name, reason in check(root, ref)]
    for ref, name, reason in failures:
        print(f"BLOCKED [{ref or 'working tree'}]: {name}: {reason}", file=sys.stderr)
    if failures:
        print("Keep personal datasets and all derived reports outside Git. Only the fictional fixture is public.", file=sys.stderr)
        return 1
    print("PASS: no recognized personal evaluation artifacts in the checked trees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
