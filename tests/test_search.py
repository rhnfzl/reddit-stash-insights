"""Tests for search orchestrator."""
import unittest
import os

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
