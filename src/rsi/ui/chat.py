"""Streamlit chat interface for reddit-stash-insights RAG.

Run with: ``streamlit run src/rsi/ui/chat.py``
"""
from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from rsi.config import Settings
from rsi.indexer.search import SearchMode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROVIDERS = ["llama-cpp", "ollama", "openai"]
_SEARCH_MODES = ["hybrid", "semantic", "keyword"]

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Reddit Stash Chat", layout="wide")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_settings() -> Settings:
    """Load settings once per session."""
    if "settings" not in st.session_state:
        st.session_state.settings = Settings.load()
    return st.session_state.settings


def _get_engine(provider: str, model: str, search_mode: str, max_history: int, db_path: Path):
    """Return a cached DirectEngine, rebuilding only when parameters change."""
    cache_key = (provider, model, search_mode, max_history, str(db_path))
    if st.session_state.get("engine_key") != cache_key:
        from rsi.chat.engine import DirectEngine
        from rsi.chat.providers.base import create_provider
        from rsi.indexer.search import SearchEngine

        search_engine = SearchEngine(db_path=db_path)
        llm = create_provider(provider=provider, model=model)
        engine = DirectEngine(
            search_engine=search_engine,
            llm=llm,
            search_mode=SearchMode(search_mode),
            max_history_turns=max_history,
        )
        st.session_state.engine = engine
        st.session_state.engine_key = cache_key
    return st.session_state.engine


def _display_sources(sources: list[dict]) -> None:
    """Render source documents inside an expander."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            sub = src.get("subreddit", "?")
            fpath = src.get("file_path", "?")
            preview = src.get("text", "")[:200].replace("\n", " ")
            st.markdown(f"**[{i}]** r/{sub} -- `{fpath}`")
            st.caption(preview)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

settings = _load_settings()

with st.sidebar:
    st.title("Settings")

    provider = st.selectbox(
        "LLM Provider",
        _PROVIDERS,
        index=_PROVIDERS.index(settings.llm_provider) if settings.llm_provider in _PROVIDERS else 0,
    )
    model = st.text_input("Model", value=settings.llm_model)
    context_docs = st.slider("Context documents", min_value=1, max_value=20, value=settings.chat_context_docs)
    search_mode = st.selectbox(
        "Search mode",
        _SEARCH_MODES,
        index=_SEARCH_MODES.index(settings.chat_search_mode) if settings.chat_search_mode in _SEARCH_MODES else 0,
    )

    if st.button("Clear history"):
        st.session_state.messages = []
        if "engine" in st.session_state:
            st.session_state.engine.clear_history()
        st.rerun()

# ---------------------------------------------------------------------------
# Initialise session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Chat display
# ---------------------------------------------------------------------------

st.title("Reddit Stash Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            _display_sources(msg["sources"])

# ---------------------------------------------------------------------------
# User input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about your Reddit archive..."):
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        try:
            engine = _get_engine(
                provider=provider,
                model=model,
                search_mode=search_mode,
                max_history=settings.chat_max_history,
                db_path=settings.db_path,
            )
            token_iter, sources = engine.chat_stream(prompt, limit=context_docs)
            answer = st.write_stream(token_iter)
            _display_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        except Exception:
            logger.exception("Chat error")
            st.error("Something went wrong. Check that your LLM provider is running and the index exists.")
