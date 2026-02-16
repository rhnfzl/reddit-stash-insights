"""Read and filter reddit-stash file_log.json entries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

FileLogEntry = Dict[str, str]
FileLog = Dict[str, FileLogEntry]


def read_file_log(path: Path) -> FileLog:
    """Read file_log.json and return all entries."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_post_entries(log: FileLog) -> FileLog:
    """Filter file_log entries to only Submission (post) entries."""
    return {k: v for k, v in log.items() if v.get("type") == "Submission"}


def get_comment_entries(log: FileLog) -> FileLog:
    """Filter file_log entries to only Comment entries."""
    return {k: v for k, v in log.items() if v.get("type") == "Comment"}
