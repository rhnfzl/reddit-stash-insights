"""Index metadata: which embedder built an index, so readers never mix vector spaces."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

META_FILE = "rsi_index.json"
SCHEMA = 1
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
) -> None:
    """Write metadata atomically.

    Call it with ``complete=False`` before the first row is written, and with
    ``complete=True`` last, after every row and the FTS index exist.
    """
    data = {
        "schema": SCHEMA,
        "complete": complete,
        "provider": provider,
        "model": model,
        "dim": dim,
        "rows": rows,
        "parse_errors": parse_errors,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = meta_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def remove_meta(db_path: Path) -> None:
    meta_path(db_path).unlink(missing_ok=True)


def check_compatible(
    db_path: Path, has_table: bool, provider: str, model: str, *, allow_incomplete: bool = False,
) -> None:
    """Raise IndexConfigMismatch if an existing index was built by another embedder.

    An absent index (no table, no metadata) is compatible with anything, so a
    fresh build works with any provider. A table without metadata is a legacy
    local BGE-M3 index. Metadata marked incomplete (a build that crashed or is
    still running) raises IndexIncomplete unless ``allow_incomplete`` (the indexer resuming it).
    """
    meta = read_meta(db_path)
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
