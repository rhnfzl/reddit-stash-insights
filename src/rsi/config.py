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
    embedding_provider: str = "local"  # local (BGE-M3 in process) or openai (OpenAI-compatible API)
    embedding_model: str = "BAAI/bge-m3"
    embedding_use_fp16: bool = True
    embedding_base_url: Optional[str] = None  # openai provider only; None = SDK default / OPENAI_BASE_URL
    embedding_batch_size: int = 128  # max inputs per embeddings request (openai)
    embedding_max_input_tokens: int = 8000  # per-input cap, below the 8191-token model limit (openai)
    embedding_max_batch_tokens: int = 250_000  # per-request cap, below the 300k API limit (openai)
    index_chunk_size: int = 512  # items embedded and inserted per chunk while indexing
    # Characters of each item kept for indexing; embedding models read about 32k anyway,
    # and long comment threads otherwise keep the whole archive in memory several times.
    max_text_chars: int = 32_000

    # Vector store
    db_path: Path = field(default_factory=lambda: DEFAULT_DB_PATH)

    # LLM (for future chat module)
    llm_provider: str = "llama-cpp"  # llama-cpp, ollama, openai
    llm_model: str = "qwen2.5:7b"

    # Chat / RAG
    chat_context_docs: int = 5  # number of context docs for RAG
    chat_search_mode: str = "hybrid"  # default search mode for chat
    chat_max_history: int = 10  # max conversation turns to keep

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
            settings.embedding_provider = embed.get("provider", settings.embedding_provider)
            settings.embedding_base_url = embed.get("base_url", settings.embedding_base_url)

            store = data.get("store", {})
            if "db_path" in store:
                settings.db_path = Path(store["db_path"])

            llm = data.get("llm", {})
            settings.llm_provider = llm.get("provider", settings.llm_provider)
            settings.llm_model = llm.get("model", settings.llm_model)

            chat = data.get("chat", {})
            settings.chat_context_docs = chat.get("context_docs", settings.chat_context_docs)
            settings.chat_search_mode = chat.get("search_mode", settings.chat_search_mode)
            settings.chat_max_history = chat.get("max_history", settings.chat_max_history)

            s3 = data.get("s3", {})
            settings.s3_bucket = s3.get("bucket", settings.s3_bucket)
            settings.s3_prefix = s3.get("prefix", settings.s3_prefix)
            if "cache_dir" in s3:
                settings.s3_cache_dir = Path(s3["cache_dir"])

        # Environment variable overrides (highest precedence)
        if v := os.environ.get("RSI_EMBEDDING_MODEL"):
            settings.embedding_model = v
        if v := os.environ.get("RSI_EMBEDDING_PROVIDER"):
            settings.embedding_provider = v
        if v := os.environ.get("RSI_EMBEDDING_BASE_URL"):
            settings.embedding_base_url = v
        if v := os.environ.get("RSI_EMBEDDING_BATCH"):
            settings.embedding_batch_size = int(v)
        if v := os.environ.get("RSI_INDEX_CHUNK"):
            settings.index_chunk_size = int(v)
        if v := os.environ.get("RSI_MAX_TEXT_CHARS"):
            settings.max_text_chars = int(v)
        if v := os.environ.get("RSI_DB_PATH"):
            settings.db_path = Path(v)
        if v := os.environ.get("RSI_LLM_PROVIDER"):
            settings.llm_provider = v
        if v := os.environ.get("RSI_LLM_MODEL"):
            settings.llm_model = v
        if v := os.environ.get("RSI_CHAT_CONTEXT_DOCS"):
            settings.chat_context_docs = int(v)
        if v := os.environ.get("RSI_CHAT_SEARCH_MODE"):
            settings.chat_search_mode = v
        if v := os.environ.get("RSI_S3_BUCKET"):
            settings.s3_bucket = v
        if v := os.environ.get("RSI_S3_PREFIX"):
            settings.s3_prefix = v

        if settings.index_chunk_size < 1:
            raise ValueError(f"RSI_INDEX_CHUNK must be positive, got {settings.index_chunk_size}")
        if settings.max_text_chars < 1:
            raise ValueError(f"RSI_MAX_TEXT_CHARS must be positive, got {settings.max_text_chars}")
        if settings.embedding_provider not in ("local", "openai"):
            raise ValueError(f"embedding provider must be 'local' or 'openai', got {settings.embedding_provider!r}")
        # The local default model name means nothing to an OpenAI-compatible API.
        if settings.embedding_provider == "openai" and settings.embedding_model == "BAAI/bge-m3":
            settings.embedding_model = "text-embedding-3-small"

        # Ensure directories exist
        settings.db_path.parent.mkdir(parents=True, exist_ok=True)

        return settings
