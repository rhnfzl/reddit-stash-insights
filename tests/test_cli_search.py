"""Tests for CLI index and search commands."""
import unittest
import tempfile
import os
from pathlib import Path
from typer.testing import CliRunner

SKIP_SLOW = os.environ.get("RSI_SKIP_SLOW_TESTS", "0") == "1"


class TestIndexCommand(unittest.TestCase):
    @unittest.skipIf(SKIP_SLOW, "Skipping slow model test")
    def test_index_command_creates_database(self):
        from rsi.cli import app

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            reddit_dir = Path(tmpdir) / "reddit"
            reddit_dir.mkdir()
            sub_dir = reddit_dir / "Python"
            sub_dir.mkdir()
            post = sub_dir / "POST_abc.md"
            post.write_text(
                "---\nid: abc\nsubreddit: /r/Python\ntimestamp: 2024-01-01 00:00:00\n"
                "author: /u/test\ncomments: 0\n"
                "permalink: https://reddit.com/r/Python/comments/abc/test/\n---\n\n"
                "# Test Post\n\n**Upvotes:** 5 | **Permalink:** [Link](https://...)\n\nBody here.\n"
            )

            db_path = Path(tmpdir) / "index"
            result = runner.invoke(app, ["index", str(reddit_dir), "--db-path", str(db_path)])
            self.assertEqual(result.exit_code, 0, msg=result.output)
            self.assertIn("Indexed", result.output)


class TestSearchCommand(unittest.TestCase):
    def test_search_without_index_shows_error(self):
        from rsi.cli import app

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nonexistent.lance"
            result = runner.invoke(app, ["search", "test query", "--db-path", str(db_path)])
            self.assertIn("0 results", result.output.lower())


if __name__ == "__main__":
    unittest.main()
