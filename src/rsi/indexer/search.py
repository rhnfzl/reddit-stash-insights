"""High-level search engine orchestrating embedder + vector store."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rsi.indexer.embedder import Embedder
from rsi.indexer.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class SearchEngine:
    """Orchestrates BGE-M3 embedding and LanceDB search."""

    def __init__(self, db_path: Path, embedder: Optional[Embedder] = None):
        self._store = VectorStore(db_path=db_path)
        self._embedder = embedder

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def index_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Embed texts and store them in the vector database."""
        embedder = self._get_embedder()
        embeddings = embedder.embed(texts)

        records = []
        for i, text in enumerate(texts):
            record = {
                "id": ids[i],
                "text": text,
                "vector": embeddings["dense"][i],
                **metadata[i],
            }
            records.append(record)

        self._store.add_records(records)
        self._store.create_fts_index()
        logger.info("Indexed %d texts", len(texts))

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
