"""Configuration defaults for reddit-stash-insights."""
from __future__ import annotations

from pathlib import Path

DEFAULT_DB_DIR = Path.home() / ".rsi" / "index"
DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = DEFAULT_DB_DIR / "vectors.lance"
