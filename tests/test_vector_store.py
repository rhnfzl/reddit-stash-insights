"""Tests for LanceDB vector store."""
import unittest
import tempfile
from pathlib import Path


class TestVectorStore(unittest.TestCase):
    def test_create_and_query_table(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust ownership and borrowing explained",
                    "vector": [0.2] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            self.assertEqual(store.count(), 2)

    def test_search_returns_results(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            results = store.vector_search([0.1] * 1024, limit=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], "abc")

    def test_fts_search(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust ownership explained",
                    "vector": [0.2] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            store.create_fts_index()
            results = store.fts_search("Python async", limit=5)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["id"], "abc")

    def test_subreddit_filter(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust tutorial",
                    "vector": [0.1] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            results = store.vector_search([0.1] * 1024, limit=10, subreddit="Python")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["subreddit"], "Python")

    def test_hybrid_search(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust ownership and borrowing explained",
                    "vector": [0.9] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            store.create_fts_index()
            results = store.hybrid_search(
                query="Python programming",
                query_vector=[0.1] * 1024,
                limit=5,
            )
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["id"], "abc")

    def test_hybrid_search_with_subreddit_filter(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust ownership and borrowing explained",
                    "vector": [0.9] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            store.create_fts_index()
            results = store.hybrid_search(
                query="programming",
                query_vector=[0.1] * 1024,
                limit=5,
                subreddit="Python",
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["subreddit"], "Python")

    def test_empty_store_returns_empty(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            self.assertEqual(store.count(), 0)
            self.assertEqual(store.vector_search([0.1] * 1024), [])
            self.assertEqual(store.fts_search("test"), [])
            self.assertEqual(
                store.hybrid_search("test", [0.1] * 1024), []
            )

    def test_append_records(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")
            store.add_records([
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
            ])
            self.assertEqual(store.count(), 1)
            store.add_records([
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust tutorial",
                    "vector": [0.2] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ])
            self.assertEqual(store.count(), 2)


if __name__ == "__main__":
    unittest.main()
