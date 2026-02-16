"""Tests for file_log reader and directory scanner."""
import unittest
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class TestFileLog(unittest.TestCase):
    def test_read_file_log(self):
        from rsi.core.file_log import read_file_log

        entries = read_file_log(FIXTURES / "sample_file_log.json")
        self.assertEqual(len(entries), 2)

    def test_file_log_entry_fields(self):
        from rsi.core.file_log import read_file_log

        entries = read_file_log(FIXTURES / "sample_file_log.json")
        post_entry = entries["1grc5bi-Python-Submission-POST"]
        self.assertEqual(post_entry["subreddit"], "Python")
        self.assertEqual(post_entry["type"], "Submission")

    def test_file_log_separates_posts_and_comments(self):
        from rsi.core.file_log import get_post_entries, get_comment_entries

        log = {
            "abc-Sub-Submission-POST": {"subreddit": "Sub", "type": "Submission", "file_path": "Sub/POST_abc.md"},
            "xyz-Sub-Comment-COMMENT": {"subreddit": "Sub", "type": "Comment", "file_path": "Sub/COMMENT_xyz.md"},
        }
        posts = get_post_entries(log)
        comments = get_comment_entries(log)
        self.assertEqual(len(posts), 1)
        self.assertEqual(len(comments), 1)


class TestScanner(unittest.TestCase):
    def test_scan_directory_finds_posts(self):
        from rsi.core.scanner import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_dir = Path(tmpdir) / "Python"
            sub_dir.mkdir()
            post = sub_dir / "POST_abc.md"
            post.write_text("---\nid: abc\nsubreddit: /r/Python\ntimestamp: 2024-01-01 00:00:00\nauthor: /u/test\ncomments: 0\npermalink: https://reddit.com/r/Python/comments/abc/test/\n---\n\n# Test Post\n\n**Upvotes:** 5 | **Permalink:** [Link](https://...)\n\nBody here.\n")

            results = scan_directory(Path(tmpdir))
            self.assertEqual(len(results.posts), 1)
            self.assertEqual(results.posts[0].id, "abc")

    def test_scan_directory_finds_comments(self):
        from rsi.core.scanner import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_dir = Path(tmpdir) / "fitbod"
            sub_dir.mkdir()
            comment = sub_dir / "COMMENT_xyz.md"
            comment.write_text("---\nComment by /u/testuser\n- **Upvotes:** 1 | **Permalink:** [Link](https://reddit.com/r/fitbod/comments/abc/test/xyz/)\nComment body here\n\n---\n\n## Context: Post by /u/other\n- **Title:** Parent Title\n- **Upvotes:** 5 | **Permalink:** [Link](https://reddit.com/r/fitbod/comments/abc/test/)\n")

            results = scan_directory(Path(tmpdir))
            self.assertEqual(len(results.comments), 1)
            self.assertEqual(results.comments[0].subreddit, "fitbod")

    def test_scan_directory_returns_scan_result(self):
        from rsi.core.scanner import scan_directory, ScanResult

        with tempfile.TemporaryDirectory() as tmpdir:
            results = scan_directory(Path(tmpdir))
            self.assertIsInstance(results, ScanResult)
            self.assertEqual(len(results.posts), 0)
            self.assertEqual(len(results.comments), 0)


if __name__ == "__main__":
    unittest.main()
