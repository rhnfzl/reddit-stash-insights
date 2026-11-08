"""LanceDB vector store for reddit-stash content."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb

logger = logging.getLogger(__name__)

TABLE_NAME = "posts"


def _subreddit_predicate(subreddit: str) -> str:
    """Keep a subreddit value inside one SQL string literal."""
    return "subreddit = '" + subreddit.replace("'", "''") + "'"


class VectorStore:
    """LanceDB-backed vector store with FTS (BM25) support."""

    def __init__(self, db_path: Path):
        self._db = lancedb.connect(str(db_path))
        self._table: Any
        try:
            self._table = self._db.open_table(TABLE_NAME)
        except Exception:
            self._table = None

    def has_table(self) -> bool:
        return self._table is not None

    def drop(self) -> None:
        """Drop the table so the next add_records starts a fresh one."""
        if self._table is not None:
            self._db.drop_table(TABLE_NAME)
            self._table = None

    def sample_texts(self, n: int) -> List[str]:
        """Text of up to ``n`` stored rows, used by the health check."""
        if self._table is None:
            return []
        rows = self._table.search().select(["text"]).limit(n).to_list()
        return [str(r["text"]) for r in rows]

    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """Upsert passages by chunk_id, or legacy rows by (content_type, id).

        Upserting makes a re-run or a retry after a failed chunk idempotent.
        """
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE_NAME)
            except Exception:
                self._table = self._db.create_table(TABLE_NAME, data=records)
                logger.info("Added %d records to vector store", len(records))
                return
        (
            self._table.merge_insert(["chunk_id"] if "chunk_id" in records[0] else ["content_type", "id"])
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(records)
        )
        logger.info("Added %d records to vector store", len(records))

    def remove_stale_chunks(self, source_doc_id: str, generation: str) -> None:
        """Remove prior document generations only after all replacement passages exist."""
        if self._table is not None:
            source = source_doc_id.replace("'", "''")
            version = generation.replace("'", "''")
            self._table.delete(f"source_doc_id = '{source}' AND generation != '{version}'")

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
            q = q.where(_subreddit_predicate(subreddit))
        result: list[dict[str, Any]] = q.to_list()
        return result

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
            q = q.where(_subreddit_predicate(subreddit))
        result: list[dict[str, Any]] = q.to_list()
        return result

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
            q = q.where(_subreddit_predicate(subreddit))
        result: list[dict[str, Any]] = q.to_list()
        return result

    def count(self) -> int:
        """Return the number of records in the store."""
        if self._table is None:
            return 0
        count: int = self._table.count_rows()
        return count
