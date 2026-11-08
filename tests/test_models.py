"""Tests for core data models."""
import unittest
from datetime import datetime


class TestPost(unittest.TestCase):
    def test_create_post_with_required_fields(self):
        from rsi.core.models import Post

        post = Post(
            id="fictionalthread",
            subreddit="synthetic",
            title="Fictional Garden Plan",
            body="Some body text",
            permalink="https://reddit.com/r/synthetic/comments/fictionalthread/fictional_garden_plan/",
            file_path="synthetic/POST_fictionalthread.md",
        )
        self.assertEqual(post.id, "fictionalthread")
        self.assertEqual(post.subreddit, "synthetic")
        self.assertEqual(post.title, "Fictional Garden Plan")

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

        ts = datetime(2020, 1, 2, 3, 4, 5)
        post = Post(
            id="fictionalthread",
            subreddit="synthetic",
            title="Fictional Garden Plan",
            body="Body text here",
            permalink="https://reddit.com/...",
            file_path="synthetic/POST_fictionalthread.md",
            author="fixture_author",
            timestamp=ts,
            flair="Showcase",
            score=12,
            comment_count=3,
        )
        self.assertEqual(post.author, "fixture_author")
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
            id="fictionalcomment",
            subreddit="synthetic",
            author="fixture_author",
            body="The blue planter belongs by the window.",
            permalink="https://reddit.com/r/synthetic/.../fictionalcomment/",
            file_path="synthetic/COMMENT_fictionalcomment.md",
        )
        self.assertEqual(comment.id, "fictionalcomment")
        self.assertEqual(comment.subreddit, "synthetic")

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
