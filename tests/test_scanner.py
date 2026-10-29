"""Tests for file_log reader and directory scanner."""
import unittest
import tempfile
from pathlib import Path

from rsi.core.scanner import scan_directory

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

    def test_file_log_ignores_failure_entries(self):
        """reddit-stash records failed items without a type; both filters drop them."""
        from rsi.core.file_log import get_comment_entries, get_post_entries

        log = {
            "abc-Sub-Submission-POST": {"subreddit": "Sub", "type": "Submission", "file_path": "Sub/POST_abc.md"},
            "SAVED_POST_zzz": {"status": "failed", "attempts": 2},
        }
        self.assertEqual(list(get_post_entries(log)), ["abc-Sub-Submission-POST"])
        self.assertEqual(get_comment_entries(log), {})


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


# Minimal copies of what reddit-stash writes (utils/save_utils.py, utils/gdpr_processor.py)
LEGACY_POST = (
    "---\nid: {id}\nsubreddit: /r/Python\ntimestamp: 2024-01-01 00:00:00\nauthor: /u/test\n"
    "flair: {flair}\ncomments: 3\npermalink: https://reddit.com/r/Python/comments/{id}/t/\n---\n\n"
    "# Post {id}\n\n**Upvotes:** 5 | **Permalink:** [Link](https://reddit.com/r/Python/comments/{id}/t/)\n\nBody.\n"
)
LEGACY_COMMENT = (
    "---\nComment by /u/someone\n- **Upvotes:** 2 | **Permalink:** "
    "[Link](https://reddit.com/r/Python/comments/abc/t/{id}/)\nComment text\n\n---\n\n"
    "## Context: Post by /u/op\n- **Title:** Parent Title\n"
)
FRONTMATTER_COMMENT = (
    '---\nid: "{id}"\ntitle: "Comment by /u/someone"\nauthor: "/u/someone"\n'
    'body: "Line one\\nline two: with colon"\nscore: 4\n'
    'permalink: "https://reddit.com/r/Python/comments/abc/t/{id}/"\n---\n\n'
    "# Comment by /u/someone\n\nLine one\nline two: with colon\n\n"
    "**Upvotes:** 4 | **Permalink:** [Link](https://reddit.com/r/Python/comments/abc/t/{id}/)\n\n"
    "## Context: Post by /u/op\n- **Title:** Parent Title\n"
)
GDPR_CSV_POST = (
    '---\nid: "{id}"\ntitle: "GDPR Export Post: {id}"\n'
    'permalink: "https://www.reddit.com/r/Python/comments/{id}/t/"\narchive_enriched: false\n---\n\n'
    "# GDPR Export Post: {id}\n\n**Reddit Link:** [link](https://www.reddit.com/r/Python/comments/{id}/t/)\n\n"
    "**Post ID:** {id}\n\n---\n\n*This is a CSV-only export (no API credentials available).*\n"
)
GDPR_CSV_COMMENT = (
    '---\nid: "{id}"\ntitle: "GDPR Export Comment: {id}"\n'
    'permalink: "https://www.reddit.com/r/Python/comments/abc/t/{id}/"\narchive_enriched: false\n---\n\n'
    "# GDPR Export Comment: {id}\n\n**Reddit Link:** [link](https://www.reddit.com/r/Python/comments/abc/t/{id}/)\n\n"
    "**Comment ID:** {id}\n\n---\n\n*This is a CSV-only export (no API credentials available).*\n"
)


class TestScannerAllFileTypes(unittest.TestCase):
    def _scan(self, files):
        from rsi.core.scanner import scan_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            for rel, text in files.items():
                path = Path(tmpdir) / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            return scan_directory(Path(tmpdir))

    def test_every_prefix_is_indexed_with_source(self):
        result = self._scan({
            "Python/POST_p1.md": LEGACY_POST.format(id="p1", flair="Showcase"),
            "Python/SAVED_POST_p2.md": LEGACY_POST.format(id="p2", flair="Showcase"),
            "Python/UPVOTE_POST_p3.md": LEGACY_POST.format(id="p3", flair="Showcase"),
            "Python/GDPR_POST_p4.md": LEGACY_POST.format(id="p4", flair="Showcase"),
            "Python/COMMENT_c1.md": LEGACY_COMMENT.format(id="c1"),
            "Python/SAVED_COMMENT_c2.md": LEGACY_COMMENT.format(id="c2"),
            "Python/UPVOTE_COMMENT_c3.md": FRONTMATTER_COMMENT.format(id="c3"),
            "Python/GDPR_COMMENT_c4.md": FRONTMATTER_COMMENT.format(id="c4"),
            "Python/notes.md": "not a reddit-stash file\n",
        })
        self.assertEqual(result.errors, [])
        self.assertEqual(
            {p.id: p.source for p in result.posts},
            {"p1": "own", "p2": "saved", "p3": "upvoted", "p4": "gdpr"},
        )
        self.assertEqual(
            {c.id: c.source for c in result.comments},
            {"c1": "own", "c2": "saved", "c3": "upvoted", "c4": "gdpr"},
        )

    def test_frontmatter_comment_uses_frontmatter_fields(self):
        result = self._scan({"Python/SAVED_COMMENT_c9.md": FRONTMATTER_COMMENT.format(id="c9")})
        comment = result.comments[0]
        self.assertEqual(comment.id, "c9")
        self.assertEqual(comment.author, "someone")
        self.assertEqual(comment.body, "Line one\nline two: with colon")
        self.assertEqual(comment.score, 4)
        self.assertEqual(comment.parent_title, "Parent Title")
        self.assertEqual(comment.subreddit, "Python")

    def test_gdpr_csv_only_files_degrade_gracefully(self):
        result = self._scan({
            "r_Python/GDPR_POST_g1.md": GDPR_CSV_POST.format(id="g1"),
            "r_Python/GDPR_COMMENT_g2.md": GDPR_CSV_COMMENT.format(id="g2"),
        })
        self.assertEqual(result.errors, [])
        post, comment = result.posts[0], result.comments[0]
        self.assertEqual((post.id, post.subreddit, post.title), ("g1", "Python", "GDPR Export Post: g1"))
        self.assertEqual(post.permalink, "https://www.reddit.com/r/Python/comments/g1/t/")
        self.assertEqual((comment.id, comment.subreddit, comment.source), ("g2", "Python", "gdpr"))
        self.assertIn("GDPR Export Comment: g2", comment.body)
        self.assertEqual(comment.author, "")

    def test_legacy_unquoted_flair_with_colons_parses(self):
        result = self._scan({"Python/UPVOTE_POST_f1.md": LEGACY_POST.format(id="f1", flair="Use: Tool: X")})
        self.assertEqual(result.errors, [])
        post = result.posts[0]
        self.assertEqual(post.flair, "Use: Tool: X")
        self.assertEqual(post.comment_count, 3)
        self.assertIsNotNone(post.timestamp)

    def test_same_item_saved_and_upvoted_is_indexed_once(self):
        result = self._scan({
            "Python/SAVED_POST_d1.md": LEGACY_POST.format(id="d1", flair="x"),
            "Python/UPVOTE_POST_d1.md": LEGACY_POST.format(id="d1", flair="x"),
        })
        self.assertEqual([(p.id, p.source) for p in result.posts], [("d1", "saved")])


if __name__ == "__main__":
    unittest.main()


class TestScannerReviewFixes(unittest.TestCase):
    """Duplicate priority, '---' inside a comment body, and ids containing '_'."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "python").mkdir()
        (self.root / "r_python").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _post(self, name, title, folder="python"):
        (self.root / folder / name).write_text(
            f"---\nid: abc\nsubreddit: /r/python\ntimestamp: 2024-01-01 00:00:00\n"
            f"author: /u/me\nflair: null\ncomments: 0\npermalink: https://reddit.com/x\n---\n\n# {title}\n\n**Upvotes:** 1\n",
            encoding="utf-8",
        )

    def test_full_post_beats_gdpr_placeholder(self):
        self._post("GDPR_POST_abc.md", "placeholder", folder="r_python")
        self._post("UPVOTE_POST_abc.md", "upvoted copy")
        self._post("POST_abc.md", "own full post")
        result = scan_directory(self.root)
        self.assertEqual(len(result.posts), 1)
        self.assertEqual(result.posts[0].title, "own full post")
        self.assertEqual(result.posts[0].source, "own")

    def test_comment_body_with_rule_is_kept_whole(self):
        (self.root / "python" / "SAVED_COMMENT_c1.md").write_text(
            "---\nid: c1\nauthor: /u/me\nbody: |\n  first part\n  ---\n  second part\n"
            "score: 3\npermalink: https://reddit.com/c1\n---\n",
            encoding="utf-8",
        )
        result = scan_directory(self.root)
        self.assertEqual(result.errors, [])
        self.assertEqual(len(result.comments), 1)
        self.assertIn("second part", result.comments[0].body)

    def test_id_with_underscore(self):
        from rsi.core.parser import item_id, item_type

        f = Path("GDPR_POST_t3_abc.md")
        self.assertEqual(item_type(f), "GDPR_POST")
        self.assertEqual(item_id(f), "t3_abc")
        self.assertEqual(item_type(Path("notes.md")), "")
