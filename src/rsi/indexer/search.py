"""High-level search engine orchestrating embedder + vector store."""
from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rsi.core.identity import canonical_doc_id
from rsi.core.models import Comment, Post
from rsi.indexer.chunker import iter_chunks
from rsi.indexer.embedder import make_embedder
from rsi.indexer.index_meta import IndexConfigMismatch, check_compatible
from rsi.indexer.vector_store import VectorStore

if TYPE_CHECKING:
    from rsi.config import Settings

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class SearchEngine:
    """Orchestrates embedding and LanceDB search.

    The embedder is ``embedder`` if given, else built lazily from ``settings``
    (``Settings.load()`` when omitted). Unless ``verify`` is False, an existing
    index built by a different embedder raises ``IndexConfigMismatch`` here,
    before any embedding call, and an unfinished build raises
    ``IndexIncomplete`` unless ``allow_incomplete`` (only the indexer sets it).
    """

    def __init__(
        self,
        db_path: Path,
        embedder: Optional[Any] = None,
        *,
        settings: Optional["Settings"] = None,
        verify: bool = True,
        allow_incomplete: bool = False,
        writing: bool = False,
    ):
        if settings is None:
            from rsi.config import Settings

            settings = Settings.load()
        self._settings = settings
        self._db_path = Path(db_path)
        self._store = VectorStore(db_path=db_path)
        self._embedder = embedder
        self._embedder_lock = threading.Lock()
        if verify:
            provider, model = self.embedder_identity()
            check_compatible(
                self._db_path, self._store.has_table(), provider, model, allow_incomplete=allow_incomplete,
                chunk_tokens=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap, writing=writing,
            )

    @property
    def store(self) -> VectorStore:
        return self._store

    def embedder_identity(self) -> tuple[str, str]:
        """(provider, model) of the embedder this engine uses, without loading it."""
        if self._embedder is not None:
            provider, model = getattr(self._embedder, "provider", None), getattr(self._embedder, "model", None)
            if isinstance(provider, str) and isinstance(model, str):
                return provider, model
        return self._settings.embedding_provider, self._settings.embedding_model

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            # Concurrent first searches must not each load a (possibly 2 GB) model.
            with self._embedder_lock:
                if self._embedder is None:
                    self._embedder = make_embedder(self._settings)
        return self._embedder

    def index_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]],
        chunk_size: Optional[int] = None,
    ) -> int:
        """Embed and insert in bounded chunks, then build the FTS index once.

        Only one chunk of vectors is held in memory at a time. Returns the
        vector dimension (0 when there was nothing to index).
        """
        chunk_size = self._settings.index_chunk_size if chunk_size is None else chunk_size
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self._store.has_table():
            raise IndexConfigMismatch("raw text indexing requires a fresh directory; use index_items for passages")
        embedder = self._get_embedder()
        dim = 0
        for start in range(0, len(texts), chunk_size):
            chunk = texts[start:start + chunk_size]
            dense = embedder.embed(chunk)["dense"]
            records = [
                {"id": ids[start + i], "text": text, "vector": dense[i], **metadata[start + i]}
                for i, text in enumerate(chunk)
            ]
            dim = len(dense[0])
            self._store.add_records(records)
            del records, dense
            logger.info("Indexed %d/%d texts", min(start + chunk_size, len(texts)), len(texts))

        if texts:
            self._store.create_fts_index()
        return dim

    def validate_index_settings(self) -> int:
        """Reject passage budgets that the active encoder would silently truncate."""
        capacity = self._get_embedder().max_input_tokens
        if type(capacity) is not int or capacity < 1:
            raise ValueError("embedding capacity must be a positive integer")
        tokens, overlap = self._settings.chunk_tokens, self._settings.chunk_overlap
        if type(tokens) is not int or type(overlap) is not int or not 0 <= overlap < tokens:
            raise ValueError("chunk tokens must be positive and 0 <= overlap < chunk tokens")
        if tokens > capacity:
            raise ValueError(f"chunk_tokens {tokens} exceeds embedding capacity {capacity}; reduce RSI_CHUNK_TOKENS")
        if self._settings.index_chunk_size < 1:
            raise ValueError("index batch size must be positive")
        return capacity

    def index_items(self, items: Iterable[Post | Comment]) -> int:
        """Stream passages and vectors in bounded batches; replace each source after embedding."""
        provider, model = self.embedder_identity()
        check_compatible(self._db_path, self._store.has_table(), provider, model, allow_incomplete=True,
                         chunk_tokens=self._settings.chunk_tokens, chunk_overlap=self._settings.chunk_overlap,
                         writing=True)
        self.validate_index_settings()
        embedder = self._get_embedder()
        batch_size = self._settings.index_chunk_size
        batch: list[dict[str, Any]] = []
        finished: list[tuple[str, str]] = []
        dim = 0

        def flush() -> None:
            nonlocal dim
            if batch:
                dense = embedder.embed([row["text"] for row in batch])["dense"]
                if len(dense) != len(batch) or not dense or not dense[0]:
                    raise RuntimeError("embedder returned an invalid passage vector batch")
                dim = len(dense[0])
                self._store.add_records([{**row, "vector": vector} for row, vector in zip(batch, dense)])
                batch.clear()
            for source, generation in finished:
                self._store.remove_stale_chunks(source, generation)
            finished.clear()

        for item in items:
            last = None
            for row in iter_chunks(item, encode=embedder.encode, decode=embedder.decode,
                                   max_tokens=self._settings.chunk_tokens, overlap=self._settings.chunk_overlap):
                batch.append(row)
                last = row
                if len(batch) >= batch_size:
                    flush()
            if last is not None:
                finished.append((last["source_doc_id"], last["generation"]))
        flush()
        if self._store.has_table():
            self._store.create_fts_index()
        return dim

    def search_candidates(
        self,
        query: str,
        limit: int = 50,
        mode: SearchMode = SearchMode.HYBRID,
        subreddit: Optional[str] = None,
        *,
        _query_vector: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch raw ranked passages so evaluation can measure actual pool depth."""
        if type(limit) is not int or limit < 1:
            raise ValueError("search limit must be a positive integer")
        if mode == SearchMode.KEYWORD:
            return self._store.fts_search(query, limit=limit, subreddit=subreddit)
        vector = _query_vector if _query_vector is not None else self._get_embedder().embed_query(query)["dense"]
        if mode == SearchMode.SEMANTIC:
            return self._store.vector_search(vector, limit=limit, subreddit=subreddit)
        return self._store.hybrid_search(query=query, query_vector=vector, limit=limit, subreddit=subreddit)

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return the best passage for each unique source thread."""
        if type(limit) is not int or limit < 1:
            raise ValueError("search limit must be a positive integer")
        vector: list[float] = []
        if mode != SearchMode.KEYWORD:
            vector = self._get_embedder().embed_query(query)["dense"]
        results: list[dict[str, Any]] = []
        for candidate_limit in (max(50, limit), max(100, 2 * limit), max(200, 4 * limit)):
            rows = self.search_candidates(query, limit=candidate_limit, mode=mode,
                                          subreddit=subreddit, _query_vector=vector)
            seen: set[str] = set()
            results = []
            for row in rows:
                identity = canonical_doc_id(row)
                if identity not in seen:
                    seen.add(identity)
                    results.append(row)
                    if len(results) >= limit:
                        return results
            if len(rows) < candidate_limit:
                break
        return results
