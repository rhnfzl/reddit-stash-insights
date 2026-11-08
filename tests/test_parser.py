"""Tests for markdown parser."""
import unittest
import tempfile
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


class TestParsePost(unittest.TestCase):
    def test_parse_post_extracts_frontmatter(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertEqual(post.id, "fictionalpost")
        self.assertEqual(post.subreddit, "Python")  # stripped /r/ prefix
        self.assertEqual(post.author, "fixture_author")  # stripped /u/ prefix
        self.assertEqual(post.flair, "Example")
        self.assertEqual(post.comment_count, 2)

    def test_parse_post_extracts_body(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertIn("Fictional Bookshelf Guide", post.body)
        self.assertIn("blue books", post.body)

    def test_parse_post_extracts_score_from_body(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertEqual(post.score, 5)

    def test_parse_post_sets_file_path(self):
        from rsi.core.parser import parse_post

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertTrue(post.file_path.endswith("sample_post.md"))

    def test_parse_post_parses_timestamp(self):
        from rsi.core.parser import parse_post
        from datetime import datetime

        post = parse_post(FIXTURES / "sample_post.md")
        self.assertEqual(post.timestamp, datetime(2020, 1, 2, 3, 4, 5))


class TestParseComment(unittest.TestCase):
    def test_parse_comment_extracts_fields(self):
        from rsi.core.parser import parse_comment

        comment = parse_comment(FIXTURES / "sample_comment.md", subreddit="synthetic")
        self.assertEqual(comment.author, "fixture_author")
        self.assertIn("blue planter", comment.body)
        self.assertEqual(comment.subreddit, "synthetic")

    def test_parse_comment_extracts_parent_title(self):
        from rsi.core.parser import parse_comment

        comment = parse_comment(FIXTURES / "sample_comment.md", subreddit="synthetic")
        self.assertEqual(comment.parent_title, "Fictional Garden Plan")

    def test_parse_comment_extracts_id_from_permalink(self):
        from rsi.core.parser import parse_comment

        comment = parse_comment(FIXTURES / "sample_comment.md", subreddit="synthetic")
        self.assertEqual(comment.id, "fictionalcomment")


class TestCommentContext(unittest.TestCase):
    def parse(self, content: str, filename: str = "COMMENT_root.md"):
        from rsi.core.parser import parse_comment
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / filename
            path.write_text(content)
            return parse_comment(path, subreddit="synthetic")

    def legacy(self) -> str:
        return ("---\n# Comment by /u/root_author\n"
                "**Upvotes:** 7 | **Permalink:** [Link](https://reddit.com/r/synthetic/comments/thread/title/root/)\n\n"
                "Primary comment.\n\n")

    def frontmatter(self) -> str:
        return ("---\nid: root\nauthor: /u/root_author\nbody: Primary comment.\nscore: 7\n"
                "permalink: https://reddit.com/r/synthetic/comments/thread/title/root/\n---\n")

    def test_legacy_export_retains_primary_metadata_and_two_context_replies(self):
        rendered = (self.legacy() + "---\n## Context: Parent Comment by /u/parent_author\n"
                    "Parent context.\n\n**Comment by /u/first**\nFirst reply.\n---\n"
                    "**Comment by /u/second**\nSecond reply.\n")
        comment = self.parse(rendered)
        self.assertEqual(comment.body, rendered.strip())
        self.assertEqual(comment.id, "root")
        self.assertEqual(comment.author, "root_author")
        self.assertEqual(comment.score, 7)
        self.assertEqual(comment.permalink, "https://reddit.com/r/synthetic/comments/thread/title/root/")

    def test_frontmatter_export_retains_full_post_and_nested_replies(self):
        rendered = ("# Comment by /u/root_author\n\nPrimary comment.\n\n"
                    "## Full Post Context:\n\n# Synthetic thread\nParent post evidence.\n\n"
                    "**Comment by /u/first**\nFirst reply.\n\n"
                    "> **Comment by /u/second**\n> Nested reply.\n")
        self.assertEqual(self.parse(self.frontmatter() + rendered).body, rendered.strip())

    def test_markdown_horizontal_rule_does_not_cut_comment_evidence(self):
        rendered = self.legacy() + "First section.\n\n---\n\nLater evidence.\n"
        self.assertEqual(self.parse(rendered).body, rendered.strip())

    def test_gdpr_archive_comment_heading_after_separator_is_retained(self):
        frontmatter = ("---\nid: root\ntitle: 'GDPR Export Comment: root'\n"
                       "permalink: https://www.reddit.com/r/synthetic/comments/thread/title/root/\n---\n")
        rendered = ("# GDPR Export Comment: root\n\n"
                    "**Reddit Link:** [link](https://www.reddit.com/r/synthetic/comments/thread/title/root/)\n\n"
                    "**Comment ID:** root\n\n---\n\n## Comment\nArchived comment evidence.\n")
        comment = self.parse(frontmatter + rendered, filename="GDPR_COMMENT_root.md")
        self.assertEqual(comment.body, rendered.strip())
        self.assertEqual(comment.source, "gdpr")
        self.assertEqual(comment.id, "root")
        self.assertEqual(comment.author, "")
        self.assertEqual(comment.permalink, "https://www.reddit.com/r/synthetic/comments/thread/title/root/")

    def test_nested_yaml_preserves_root_metadata_and_parent_title(self):
        rendered = ("# Comment by /u/root_author\n\nPrimary comment.\n\n"
                    "## Context: Post by /u/parent_author\n- **Title:** Synthetic parent title\n\n"
                    "---\nid: nested_parent\nauthor: /u/nested_author\nscore: 99\n"
                    "permalink: https://reddit.com/r/other/comments/nested/title/\n---\n"
                    "# Nested parent post\nNested evidence.\n\n## Child Comments:\nChild evidence.\n")
        comment = self.parse(self.frontmatter() + rendered)
        self.assertEqual(comment.body, rendered.strip())
        self.assertEqual((comment.id, comment.author, comment.score, comment.subreddit),
                         ("root", "root_author", 7, "synthetic"))
        self.assertEqual(comment.permalink, "https://reddit.com/r/synthetic/comments/thread/title/root/")
        self.assertEqual(comment.parent_title, "Synthetic parent title")

    def test_frontmatter_body_remains_fallback_for_empty_rendered_markdown(self):
        self.assertEqual(self.parse(self.frontmatter() + "\n \n").body, "Primary comment.")


if __name__ == "__main__":
    unittest.main()
