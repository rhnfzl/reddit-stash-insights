"""LanceDB vector store for reddit-stash content."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb

logger = logging.getLogger(__name__)

TABLE_NAME = "posts"


class VectorStore:
    """LanceDB-backed vector store with FTS (BM25) support."""

    def __init__(self, db_path: Path):
        self._db = lancedb.connect(str(db_path))
        self._table = None

    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """Add records to the vector store. Creates table on first call, appends after."""
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE_NAME)
                self._table.add(records)
            except Exception:
                self._table = self._db.create_table(TABLE_NAME, data=records)
        else:
            self._table.add(records)
        logger.info("Added %d records to vector store", len(records))

    def create_fts_index(self) -> None:
        """Create a Tantivy full-text search index on the 'text' column."""
        if self._table is None:
            raise RuntimeError("No table exists yet. Add records first.")
        self._table.create_fts_index("text", replace=True)
        logger.info("Created FTS index on 'text' column")

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Pure vector similarity search."""
        if self._table is None:
            return []
        q = self._table.search(query_vector).limit(limit)
        if subreddit:
            q = q.where(f"subreddit = '{subreddit}'")
        return q.to_list()

    def fts_search(
        self,
        query: str,
        limit: int = 10,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text (BM25) keyword search via Tantivy."""
        if self._table is None:
            return []
        q = self._table.search(query, query_type="fts").limit(limit)
        if subreddit:
            q = q.where(f"subreddit = '{subreddit}'")
        return q.to_list()

    def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 10,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 + vector similarity with RRF reranking.

        Uses LanceHybridQueryBuilder with explicit text() and vector() calls,
        which is required when no embedding function is registered on the table.
        """
        from lancedb.query import LanceHybridQueryBuilder
        from lancedb.rerankers import RRFReranker

        if self._table is None:
            return []

        reranker = RRFReranker(K=60)
        q = (
            LanceHybridQueryBuilder(self._table)
            .vector(query_vector)
            .text(query)
            .rerank(reranker=reranker)
            .limit(limit)
        )
        if subreddit:
            q = q.where(f"subreddit = '{subreddit}'")
        return q.to_list()

    def count(self) -> int:
        """Return the number of records in the store."""
        if self._table is None:
            return 0
        return self._table.count_rows()
