"""Tests for core data models."""
import unittest
from datetime import datetime


class TestPost(unittest.TestCase):
    def test_create_post_with_required_fields(self):
        from rsi.core.models import Post

        post = Post(
            id="1njd7m5",
            subreddit="fitbod",
            title="Fitbod Screen Stuck",
            body="Some body text",
            permalink="https://reddit.com/r/fitbod/comments/1njd7m5/fitbod_screen_stuck/",
            file_path="fitbod/POST_1njd7m5.md",
        )
        self.assertEqual(post.id, "1njd7m5")
        self.assertEqual(post.subreddit, "fitbod")
        self.assertEqual(post.title, "Fitbod Screen Stuck")

    def test_post_optional_fields_default_to_none(self):
        from rsi.core.models import Post

        post = Post(
            id="abc",
            subreddit="test",
            title="Test",
            body="",
            permalink="https://reddit.com/...",
            file_path="test/POST_abc.md",
        )
        self.assertIsNone(post.author)
        self.assertIsNone(post.timestamp)
        self.assertIsNone(post.flair)
        self.assertEqual(post.score, 0)
        self.assertEqual(post.comment_count, 0)

    def test_post_with_all_fields(self):
        from rsi.core.models import Post

        ts = datetime(2025, 9, 17, 13, 29, 49)
        post = Post(
            id="1njd7m5",
            subreddit="fitbod",
            title="Fitbod Screen Stuck",
            body="Body text here",
            permalink="https://reddit.com/...",
            file_path="fitbod/POST_1njd7m5.md",
            author="complexrexton",
            timestamp=ts,
            flair="Showcase",
            score=12,
            comment_count=3,
        )
        self.assertEqual(post.author, "complexrexton")
        self.assertEqual(post.timestamp, ts)
        self.assertEqual(post.flair, "Showcase")
        self.assertEqual(post.score, 12)
        self.assertEqual(post.comment_count, 3)

    def test_post_search_text_combines_title_and_body(self):
        from rsi.core.models import Post

        post = Post(
            id="x", subreddit="s", title="My Title",
            body="The body content here",
            permalink="https://...", file_path="s/POST_x.md",
        )
        text = post.search_text()
        self.assertIn("My Title", text)
        self.assertIn("The body content here", text)


class TestComment(unittest.TestCase):
    def test_create_comment(self):
        from rsi.core.models import Comment

        comment = Comment(
            id="nepvcs7",
            subreddit="fitbod",
            author="complexrexton",
            body="Thank you it's working now",
            permalink="https://reddit.com/r/fitbod/.../nepvcs7/",
            file_path="fitbod/COMMENT_nepvcs7.md",
        )
        self.assertEqual(comment.id, "nepvcs7")
        self.assertEqual(comment.subreddit, "fitbod")

    def test_comment_search_text(self):
        from rsi.core.models import Comment

        comment = Comment(
            id="x", subreddit="s", author="u",
            body="This is the comment body",
            permalink="https://...", file_path="s/COMMENT_x.md",
            parent_title="Parent Post Title",
        )
        text = comment.search_text()
        self.assertIn("Parent Post Title", text)
        self.assertIn("This is the comment body", text)


if __name__ == "__main__":
    unittest.main()
