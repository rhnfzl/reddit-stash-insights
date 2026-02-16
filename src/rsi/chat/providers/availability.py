"""Provider availability checking and auto-fallback.

Pre-flights each LLM provider before attempting to create it, and falls back
to alternatives when the selected provider is unavailable.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS_DIR = Path.home() / ".rsi" / "models"

# Ordered fallback preference: local-first, then remote.
_FALLBACK_ORDER = ["llama-cpp", "ollama", "openai"]


def check_provider(provider: str, model: str) -> str | None:
    """Check whether *provider* is available right now.

    Returns ``None`` if the provider is ready, or a human-readable error
    string explaining why it is not.
    """
    name = provider.lower().strip()

    if name == "ollama":
        return _check_ollama()
    if name == "llama-cpp":
        return _check_llama_cpp(model)
    if name == "openai":
        return _check_openai()

    return f"Unknown provider: {provider!r}"


def find_available_provider(
    selected: str,
    model: str,
) -> tuple[str | None, str | None, str | None]:
    """Try *selected* provider first, then iterate fallbacks.

    Returns ``(provider, model_to_use, fallback_note | None)``.
    If no provider is available, returns ``(None, None, error_message)``.
    """
    # 1. Try the selected provider.
    err = check_provider(selected, model)
    if err is None:
        resolved_model = _resolve_model(selected, model)
        return (selected, resolved_model, None)

    logger.warning("Selected provider %r unavailable: %s", selected, err)
    errors = [f"{selected}: {err}"]

    # 2. Try alternatives in fallback order.
    for alt in _FALLBACK_ORDER:
        if alt == selected:
            continue
        alt_model = _resolve_model(alt, model)
        alt_err = check_provider(alt, alt_model)
        if alt_err is None:
            note = f"Using {alt} ({selected} not available: {err})"
            logger.info(note)
            return (alt, alt_model, note)
        errors.append(f"{alt}: {alt_err}")

    all_errors = "; ".join(errors)
    return (None, None, f"No LLM provider available. {all_errors}")


# ---------------------------------------------------------------------------
# Per-provider checks
# ---------------------------------------------------------------------------


def _check_ollama() -> str | None:
    """Ping the Ollama HTTP server."""
    try:
        import requests
    except ImportError:
        return "requests package not installed"

    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return f"Ollama server not reachable: {exc}"
    return None


def _check_llama_cpp(model: str) -> str | None:
    """Verify llama-cpp-python is importable and a GGUF model file exists."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return "llama-cpp-python package not installed"

    resolved = _resolve_gguf_path(model)
    if resolved is None:
        return f"No GGUF model found (checked {model!r} and {_MODELS_DIR})"
    return None


def _check_openai() -> str | None:
    """Verify the openai package is importable and an API key is set."""
    try:
        import openai  # noqa: F401
    except ImportError:
        return "openai package not installed"

    if not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY environment variable not set"
    return None


# ---------------------------------------------------------------------------
# Model resolution helpers
# ---------------------------------------------------------------------------


def _resolve_gguf_path(model: str) -> str | None:
    """Return an existing GGUF file path, or ``None``.

    If *model* is already a valid path to a .gguf file, return it.
    Otherwise, scan ``~/.rsi/models/`` for the first .gguf file.
    """
    p = Path(model).expanduser()
    if p.suffix.lower() == ".gguf" and p.exists():
        return str(p)

    # Auto-detect from models directory.
    if _MODELS_DIR.is_dir():
        # Sort for deterministic results.
        for gguf in sorted(_MODELS_DIR.glob("*.gguf")):
            return str(gguf)
        # Also check split GGUF (e.g. model-00001-of-00003.gguf)
        for gguf in sorted(_MODELS_DIR.glob("*-00001-of-*.gguf")):
            return str(gguf)
    return None


def _resolve_model(provider: str, model: str) -> str:
    """Adjust the model value when switching providers.

    For ``llama-cpp``, ensure *model* points to an actual GGUF file.
    For other providers, return *model* unchanged.
    """
    if provider == "llama-cpp":
        resolved = _resolve_gguf_path(model)
        if resolved:
            return resolved
    return model
