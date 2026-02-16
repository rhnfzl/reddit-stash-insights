"""Tests for markdown parser."""
import unittest
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class TestParsePost(unittest.TestCase):
    def test_parse_post_extracts_frontmatter(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertEqual(post.id, "1grc5bi")
        self.assertEqual(post.subreddit, "Python")  # stripped /r/ prefix
        self.assertEqual(post.author, "complexrexton")  # stripped /u/ prefix
        self.assertEqual(post.flair, "Showcase")
        self.assertEqual(post.comment_count, 7)

    def test_parse_post_extracts_body(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertIn("SqueakyCleanText", post.body)
        self.assertIn("text preprocessing", post.body)

    def test_parse_post_extracts_score_from_body(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertEqual(post.score, 12)

    def test_parse_post_sets_file_path(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertTrue(post.file_path.endswith("sample_post.md"))

    def test_parse_post_parses_timestamp(self):
        from rsi.core.parser import parse_post
        from datetime import datetime

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertEqual(post.timestamp, datetime(2024, 11, 14, 18, 52, 30))


class TestParseComment(unittest.TestCase):
    def test_parse_comment_extracts_fields(self):
        from rsi.core.parser import parse_comment

        comment = parse_comment(FIXTURES / "sample_comment.md", subreddit="fitbod")
        self.assertEqual(comment.author, "complexrexton")
        self.assertIn("working now", comment.body)
        self.assertEqual(comment.subreddit, "fitbod")

    def test_parse_comment_extracts_parent_title(self):
        from rsi.core.parser import parse_comment

        comment = parse_comment(FIXTURES / "sample_comment.md", subreddit="fitbod")
        self.assertEqual(comment.parent_title, "Fitbod Screen Stuck")

    def test_parse_comment_extracts_id_from_permalink(self):
        from rsi.core.parser import parse_comment

        comment = parse_comment(FIXTURES / "sample_comment.md", subreddit="fitbod")
        self.assertEqual(comment.id, "nepvcs7")


if __name__ == "__main__":
    unittest.main()
