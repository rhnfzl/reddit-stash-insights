"""Index metadata: which embedder built an index, so readers never mix vector spaces."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

META_FILE = "rsi_index.json"
SCHEMA = 2
# Indexes built before metadata existed were always local BGE-M3.
LEGACY_IDENTITY = ("local", "BAAI/bge-m3")


class IndexConfigMismatch(RuntimeError):
    """The index was built with a different embedder than the one configured."""


class IndexIncomplete(RuntimeError):
    """An index build started but never finished, so rows or the FTS index may be missing."""


def meta_path(db_path: Path) -> Path:
    return Path(db_path) / META_FILE


def read_meta(db_path: Path) -> Optional[Dict[str, Any]]:
    path = meta_path(db_path)
    if not path.exists():
        return None
    data: Dict[str, Any] = json.loads(path.read_text())
    return data


def write_meta(
    db_path: Path, *, provider: str, model: str, dim: int, rows: int, parse_errors: int, complete: bool = True,
    chunk_tokens: int = 512, chunk_overlap: int = 64, input_token_capacity: Optional[int] = None,
) -> None:
    """Write metadata atomically.

    Call it with ``complete=False`` before the first row is written, and with
    ``complete=True`` last, after every row and the FTS index exist.
    """
    data = {
        "schema": SCHEMA,
        "chunks": {"version": 1, "max_tokens": chunk_tokens, "overlap": chunk_overlap},
        "complete": complete,
        "provider": provider,
        "model": model,
        "dim": dim,
        "rows": rows,
        "parse_errors": parse_errors,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if input_token_capacity is not None:
        data["input_token_capacity"] = input_token_capacity
    path = meta_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def remove_meta(db_path: Path) -> None:
    meta_path(db_path).unlink(missing_ok=True)


def check_compatible(
    db_path: Path, has_table: bool, provider: str, model: str, *, allow_incomplete: bool = False,
    chunk_tokens: int = 512, chunk_overlap: int = 64, writing: bool = False,
) -> None:
    """Raise IndexConfigMismatch if an existing index was built by another embedder.

    An absent index (no table, no metadata) is compatible with anything, so a
    fresh build works with any provider. A table without metadata is a legacy
    local BGE-M3 index. Metadata marked incomplete (a build that crashed or is
    still running) raises IndexIncomplete unless ``allow_incomplete`` (the indexer resuming it).
    """
    meta = read_meta(db_path)
    if meta is not None and meta.get("schema") not in (1, SCHEMA):
        raise IndexConfigMismatch(f"unsupported index schema at {db_path}: use a fresh directory or --rebuild")
    if writing and has_table and (meta is None or meta.get("schema") != SCHEMA):
        raise IndexConfigMismatch(f"legacy index at {db_path} is read-only: use a fresh directory or --rebuild")
    expected = {"version": 1, "max_tokens": chunk_tokens, "overlap": chunk_overlap}
    if meta is not None and meta.get("schema") == SCHEMA and meta.get("chunks") != expected:
        raise IndexConfigMismatch(f"index chunk settings differ at {db_path}: use a fresh directory or --rebuild")
    if meta is not None and meta.get("complete") is False and not allow_incomplete:
        raise IndexIncomplete(
            f"index at {db_path} is an unfinished build: run 'rsi index <reddit_dir> --db-path {db_path}' again"
        )
    if meta is not None:
        built = (meta.get("provider"), meta.get("model"))
    elif has_table:
        built = LEGACY_IDENTITY
    else:
        return
    if built != (provider, model):
        raise IndexConfigMismatch(
            f"index at {db_path} was built with {built[0]}/{built[1]}, but {provider}/{model} is configured: "
            f"run 'rsi index <reddit_dir> --db-path {db_path} --rebuild'"
        )
