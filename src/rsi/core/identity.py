"""Canonical source-thread identity shared by retrieval and evaluation."""
from __future__ import annotations

import re
from typing import Any

_DOC_ID = re.compile(r"^(post|comment):[a-zA-Z0-9_-]+$")
_THREAD = re.compile(r"/comments/([a-zA-Z0-9_-]+)(?:/|[?#]|$)")


def _valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(_DOC_ID.fullmatch(value))


def canonical_doc_id(row: dict[str, Any]) -> str:
    """Resolve chunks and comment permalinks to their original source thread."""
    parent = row.get("parent_doc_id")
    if parent:
        if not _valid_id(parent):
            raise ValueError(f"invalid parent document ID: {parent!r}")
        return str(parent)
    match = _THREAD.search(str(row.get("permalink", "")))
    if match:
        return f"post:{match[1]}"
    kind, item_id = row.get("content_type"), row.get("id")
    result = f"{kind}:{item_id}"
    if not item_id or not _valid_id(result):
        raise ValueError(f"invalid ranking document ID: {result!r}")
    return result
