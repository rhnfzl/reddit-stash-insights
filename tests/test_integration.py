"""Integration test: full pipeline against real reddit-stash data.

Requires: ../reddit-stash/reddit/ directory with actual content.
Skip with: RSI_SKIP_SLOW_TESTS=1
"""
import unittest
import os
import tempfile
from pathlib import Path

SKIP_SLOW = os.environ.get("RSI_SKIP_SLOW_TESTS", "0") == "1"
REDDIT_DIR = Path(__file__).parent.parent.parent / "reddit-stash" / "reddit"


@unittest.skipIf(SKIP_SLOW, "Skipping slow integration test")
@unittest.skipUnless(REDDIT_DIR.exists(), f"reddit-stash data not found at {REDDIT_DIR}")
class TestIntegration(unittest.TestCase):
    def test_scan_real_data(self):
        from rsi.core.scanner import scan_directory

        result = scan_directory(REDDIT_DIR)
        self.assertGreater(len(result.posts), 0, "Should find at least one post")
        self.assertLess(
            len(result.errors), len(result.posts),
            "More errors than posts -- parser likely broken"
        )

    def test_full_index_and_search(self):
        from rsi.core.scanner import scan_directory
        from rsi.indexer.search import SearchEngine

        scan_result = scan_directory(REDDIT_DIR)
        self.assertGreater(len(scan_result.posts), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.lance"
            engine = SearchEngine(db_path=db_path)

            posts = scan_result.posts[:5]
            texts = [p.search_text() for p in posts]
            ids = [p.id for p in posts]
            metadata = [
                {"subreddit": p.subreddit, "score": p.score,
                 "timestamp": str(p.timestamp or ""),
                 "file_path": p.file_path, "content_type": "post"}
                for p in posts
            ]

            engine.index_texts(texts=texts, ids=ids, metadata=metadata)

            first_title = posts[0].title
            if first_title:
                results = engine.search(first_title, limit=3)
                self.assertGreater(len(results), 0, f"Should find results for: {first_title}")


if __name__ == "__main__":
    unittest.main()
