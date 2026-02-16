"""Configuration for reddit-stash-insights.

Settings loaded from (in order of precedence):
1. Environment variables (RSI_EMBEDDING_MODEL, RSI_DB_PATH, etc.)
2. Config file (~/.rsi/config.toml)
3. Defaults
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Try tomllib (3.11+), fall back to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]


RSI_DIR = Path.home() / ".rsi"
RSI_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG_PATH = RSI_DIR / "config.toml"
DEFAULT_DB_PATH = RSI_DIR / "index" / "vectors.lance"


@dataclass
class Settings:
    """Application settings — all model names and paths are configurable."""

    # Embedding
    embedding_model: str = "BAAI/bge-m3"
    embedding_use_fp16: bool = True

    # Vector store
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    # LLM (for future chat module)
    llm_provider: str = "ollama"  # ollama, llama-cpp, openai, anthropic, google
    llm_model: str = "qwen2.5:7b"

    # S3 (optional, for remote reddit-stash data)
    s3_bucket: Optional[str] = None
    s3_prefix: str = "reddit/"
    s3_cache_dir: Path = field(default_factory=lambda: RSI_DIR / "cache" / "s3")

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from config file + environment variable overrides."""
        settings = cls()

        # Load from TOML config file
        path = config_path or DEFAULT_CONFIG_PATH
        if path.exists() and tomllib is not None:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            embed = data.get("embedding", {})
            settings.embedding_model = embed.get("model", settings.embedding_model)
            settings.embedding_use_fp16 = embed.get("use_fp16", settings.embedding_use_fp16)

            store = data.get("store", {})
            if "db_path" in store:
                settings.db_path = Path(store["db_path"])

            llm = data.get("llm", {})
            settings.llm_provider = llm.get("provider", settings.llm_provider)
            settings.llm_model = llm.get("model", settings.llm_model)

            s3 = data.get("s3", {})
            settings.s3_bucket = s3.get("bucket", settings.s3_bucket)
            settings.s3_prefix = s3.get("prefix", settings.s3_prefix)
            if "cache_dir" in s3:
                settings.s3_cache_dir = Path(s3["cache_dir"])

        # Environment variable overrides (highest precedence)
        if v := os.environ.get("RSI_EMBEDDING_MODEL"):
            settings.embedding_model = v
        if v := os.environ.get("RSI_DB_PATH"):
            settings.db_path = Path(v)
        if v := os.environ.get("RSI_LLM_PROVIDER"):
            settings.llm_provider = v
        if v := os.environ.get("RSI_LLM_MODEL"):
            settings.llm_model = v
        if v := os.environ.get("RSI_S3_BUCKET"):
            settings.s3_bucket = v
        if v := os.environ.get("RSI_S3_PREFIX"):
            settings.s3_prefix = v

        # Ensure directories exist
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)

        return settings
