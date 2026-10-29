"""High-level search engine orchestrating embedder + vector store."""
from __future__ import annotations

import logging
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from rsi.indexer.embedder import make_embedder
from rsi.indexer.index_meta import check_compatible
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

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search indexed content."""
        if mode == SearchMode.KEYWORD:
            return self._store.fts_search(query, limit=limit, subreddit=subreddit)

        embedder = self._get_embedder()
        query_emb = embedder.embed_query(query)

        if mode == SearchMode.SEMANTIC:
            return self._store.vector_search(
                query_emb["dense"], limit=limit, subreddit=subreddit
            )

        # Default: hybrid
        return self._store.hybrid_search(
            query=query,
            query_vector=query_emb["dense"],
            limit=limit,
            subreddit=subreddit,
        )
