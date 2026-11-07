"""Tests for search orchestrator."""
import unittest
import os
from typing import Any

SKIP_SLOW = os.environ.get("RSI_SKIP_SLOW_TESTS", "0") == "1"


class TestSearchOrchestrator(unittest.TestCase):
    @unittest.skipIf(SKIP_SLOW, "Skipping slow model test")
    def test_search_returns_results(self):
        import tempfile
        from pathlib import Path
        from rsi.indexer.search import SearchEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SearchEngine(db_path=Path(tmpdir) / "test.lance")
            engine.index_texts(
                texts=["Python async await tutorial", "Rust memory safety"],
                ids=["abc", "def"],
                metadata=[
                    {
                        "subreddit": "Python",
                        "score": 10,
                        "timestamp": "2024-01-01",
                        "file_path": "Python/POST_abc.md",
                        "content_type": "post",
                    },
                    {
                        "subreddit": "rust",
                        "score": 25,
                        "timestamp": "2024-02-01",
                        "file_path": "rust/POST_def.md",
                        "content_type": "post",
                    },
                ],
            )
            results = engine.search("Python async", limit=2)
            self.assertGreater(len(results), 0)

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model test")
    def test_keyword_search(self):
        import tempfile
        from pathlib import Path
        from rsi.indexer.search import SearchEngine, SearchMode

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SearchEngine(db_path=Path(tmpdir) / "test.lance")
            engine.index_texts(
                texts=["Python async await tutorial", "Rust memory safety"],
                ids=["abc", "def"],
                metadata=[
                    {
                        "subreddit": "Python",
                        "score": 10,
                        "timestamp": "2024-01-01",
                        "file_path": "Python/POST_abc.md",
                        "content_type": "post",
                    },
                    {
                        "subreddit": "rust",
                        "score": 25,
                        "timestamp": "2024-02-01",
                        "file_path": "rust/POST_def.md",
                        "content_type": "post",
                    },
                ],
            )
            results = engine.search("Python", limit=2, mode=SearchMode.KEYWORD)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["id"], "abc")

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model test")
    def test_semantic_search(self):
        import tempfile
        from pathlib import Path
        from rsi.indexer.search import SearchEngine, SearchMode

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SearchEngine(db_path=Path(tmpdir) / "test.lance")
            engine.index_texts(
                texts=["Python async await tutorial", "Rust memory safety"],
                ids=["abc", "def"],
                metadata=[
                    {
                        "subreddit": "Python",
                        "score": 10,
                        "timestamp": "2024-01-01",
                        "file_path": "Python/POST_abc.md",
                        "content_type": "post",
                    },
                    {
                        "subreddit": "rust",
                        "score": 25,
                        "timestamp": "2024-02-01",
                        "file_path": "rust/POST_def.md",
                        "content_type": "post",
                    },
                ],
            )
            results = engine.search(
                "asynchronous programming in Python",
                limit=2,
                mode=SearchMode.SEMANTIC,
            )
            self.assertGreater(len(results), 0)

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model test")
    def test_subreddit_filter(self):
        import tempfile
        from pathlib import Path
        from rsi.indexer.search import SearchEngine, SearchMode

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SearchEngine(db_path=Path(tmpdir) / "test.lance")
            engine.index_texts(
                texts=["Python async await tutorial", "Rust memory safety"],
                ids=["abc", "def"],
                metadata=[
                    {
                        "subreddit": "Python",
                        "score": 10,
                        "timestamp": "2024-01-01",
                        "file_path": "Python/POST_abc.md",
                        "content_type": "post",
                    },
                    {
                        "subreddit": "rust",
                        "score": 25,
                        "timestamp": "2024-02-01",
                        "file_path": "rust/POST_def.md",
                        "content_type": "post",
                    },
                ],
            )
            results = engine.search(
                "programming tutorial",
                limit=10,
                mode=SearchMode.KEYWORD,
                subreddit="Python",
            )
            # Should only return Python subreddit results
            for r in results:
                self.assertEqual(r["subreddit"], "Python")


class TestSearchMode(unittest.TestCase):
    def test_search_mode_enum(self):
        from rsi.indexer.search import SearchMode

        self.assertEqual(SearchMode.HYBRID.value, "hybrid")
        self.assertEqual(SearchMode.SEMANTIC.value, "semantic")
        self.assertEqual(SearchMode.KEYWORD.value, "keyword")

    def test_search_mode_from_string(self):
        from rsi.indexer.search import SearchMode

        self.assertEqual(SearchMode("hybrid"), SearchMode.HYBRID)
        self.assertEqual(SearchMode("semantic"), SearchMode.SEMANTIC)
        self.assertEqual(SearchMode("keyword"), SearchMode.KEYWORD)


if __name__ == "__main__":
    unittest.main()

class TestUniqueThreads(unittest.TestCase):
    def test_all_modes_overfetch_duplicates_and_preserve_best_passage(self):
        from unittest.mock import Mock, patch
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine, SearchMode
        rows = [{'id': 'a', 'content_type': 'post', 'parent_doc_id': 'post:a', 'text': f'passage {i}'}
                for i in range(10)]
        rows += [{'id': 'c', 'content_type': 'comment', 'permalink': '/r/x/comments/a/title/c/'},
                 {'id': 'b', 'content_type': 'post', 'text': 'other thread'}]
        engine = object.__new__(SearchEngine)
        engine._settings = Settings()
        engine._store = Mock()
        embedder = Mock(embed_query=Mock(return_value={"dense": [0.1]}))
        with patch.object(engine, "_get_embedder", return_value=embedder):
            for mode, method in [(SearchMode.KEYWORD, 'fts_search'), (SearchMode.SEMANTIC, 'vector_search'),
                                 (SearchMode.HYBRID, 'hybrid_search')]:
                with self.subTest(mode=mode):
                    search = getattr(engine._store, method)
                    search.return_value = rows
                    results = engine.search('query', limit=2, mode=mode)
                    self.assertEqual([r['id'] for r in results], ['a', 'b'])
                    self.assertEqual(results[0]['text'], 'passage 0')
                    self.assertEqual(search.call_args.kwargs['limit'], 50)


class TestBoundedOverfetch(unittest.TestCase):
    def test_expands_to_100_then_200_and_stops(self):
        from unittest.mock import Mock
        from rsi.indexer.search import SearchEngine, SearchMode
        engine = object.__new__(SearchEngine)
        engine._store = Mock()
        duplicate = {"id": "p", "content_type": "post", "text": "best passage"}
        def candidates(query, *, limit, subreddit):
            return [duplicate] * limit
        engine._store.fts_search.side_effect = candidates
        self.assertEqual(engine.search("query", limit=10, mode=SearchMode.KEYWORD), [duplicate])
        self.assertEqual([call.kwargs["limit"] for call in engine._store.fts_search.call_args_list], [50, 100, 200])

class TestLegacyWriteBoundary(unittest.TestCase):
    def test_raw_text_writer_refuses_existing_index(self):
        from unittest.mock import Mock, patch
        from rsi.config import Settings
        from rsi.indexer.index_meta import IndexConfigMismatch
        from rsi.indexer.search import SearchEngine
        engine = object.__new__(SearchEngine)
        engine._settings = Settings()
        engine._store = Mock(has_table=Mock(return_value=True))
        with patch.object(engine, "_get_embedder") as get_embedder:
            with self.assertRaises(IndexConfigMismatch):
                engine.index_texts(['new text'], ['p'], [{'content_type': 'post'}])
            get_embedder.assert_not_called()

class TestRequestedLimit(unittest.TestCase):
    def test_three_hundred_unique_threads_are_returned(self):
        from unittest.mock import Mock
        from rsi.indexer.search import SearchEngine, SearchMode
        engine = object.__new__(SearchEngine)
        engine._store = Mock()
        rows = [{"id": f"p{i}", "content_type": "post"} for i in range(300)]
        engine._store.fts_search.side_effect = lambda query, *, limit, subreddit: rows[:limit]
        self.assertEqual(len(engine.search("query", limit=300, mode=SearchMode.KEYWORD)), 300)

    def test_limit_requires_positive_integer(self):
        from unittest.mock import Mock
        from rsi.indexer.search import SearchEngine, SearchMode
        engine = object.__new__(SearchEngine)
        engine._store = Mock()
        invalid_limits: tuple[Any, ...] = (0, -1, True, 1.5)
        for limit in invalid_limits:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                engine.search("query", limit=limit, mode=SearchMode.KEYWORD)
        engine._store.fts_search.assert_not_called()
