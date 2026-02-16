# Phase 1: Foundation & Semantic Search — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the core parser, data models, BGE-M3 embedding pipeline, LanceDB vector store, and hybrid search CLI — the foundation everything else builds on.

**Architecture:** A `src/rsi/` Python package with `core/` (always installed) and `indexer/` (optional `[search]` dep group). The core parser reads reddit-stash markdown files + `file_log.json`, producing `Post` and `Comment` dataclasses. The indexer embeds these with BGE-M3 (dense + sparse vectors), stores them in LanceDB with Tantivy FTS, and exposes hybrid search (BM25 + semantic + sparse) via a Typer CLI.

**Tech Stack:** Python 3.10+, PyYAML, Typer, BGE-M3 (FlagEmbedding), LanceDB, sentence-transformers

**Test framework:** `unittest` (matching reddit-stash convention). Run with `python -m unittest`.

**Input data location:** `reddit-stash/reddit/` directory (sibling of this repo). Contains:
- `{subreddit}/POST_{id}.md` — YAML frontmatter (id, subreddit, timestamp, author, flair?, comments, permalink) + markdown body
- `{subreddit}/COMMENT_{id}.md` — No YAML frontmatter. Inline markdown with author, upvotes, permalink in first lines
- `file_log.json` — `{"key": {"subreddit": "...", "type": "Submission"|"Comment", "file_path": "..."}}`

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/rsi/__init__.py`
- Create: `src/rsi/core/__init__.py`
- Create: `.gitignore`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "reddit-stash-insights"
version = "0.1.0"
description = "Semantic search, knowledge graph, analytics, and AI chat for reddit-stash archives"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "pyyaml>=6.0",
    "typer>=0.12",
]

[project.optional-dependencies]
search = [
    "sentence-transformers>=3.0",
    "lancedb>=0.29",
    "FlagEmbedding>=1.2",
]
graph = [
    "networkx>=3.0",
    "bertopic>=0.16",
    "pyvis>=0.3",
    "sentence-transformers>=3.0",
]
analytics = [
    "plotly>=5.0",
    "pandas>=2.0",
]
chat = [
    "llama-index>=0.11",
]
chat-local = [
    "llama-cpp-python>=0.3",
]
chat-ollama = [
    "ollama>=0.3",
]
mcp = [
    "mcp>=1.26",
]
ui = [
    "streamlit>=1.35",
    "plotly>=5.0",
]
all = [
    "reddit-stash-insights[search,graph,analytics,chat,chat-ollama,mcp,ui]",
]
dev = [
    "ruff>=0.5",
]

[project.scripts]
rsi = "rsi.cli:app"

[tool.ruff]
target-version = "py310"
line-length = 120
```

**Step 2: Create package init files**

`src/rsi/__init__.py`:
```python
"""reddit-stash-insights: Semantic search, knowledge graph, and AI chat for reddit-stash archives."""

__version__ = "0.1.0"
```

`src/rsi/core/__init__.py`:
```python
"""Core parser and data models — always installed."""
```

`tests/__init__.py`: empty file.

**Step 3: Create .gitignore**

```gitignore
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg
.venv/
venv/
.env
*.lance/
data/
.DS_Store
.ruff_cache/
.pytest_cache/
```

**Step 4: Commit**

```bash
git add pyproject.toml src/ tests/__init__.py .gitignore
git commit -m "feat: scaffold project structure with pyproject.toml and dependency groups"
```

---

## Task 2: Data Models

**Files:**
- Create: `src/rsi/core/models.py`
- Create: `tests/test_models.py`

**Step 1: Write the failing test**

`tests/test_models.py`:
```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_models -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rsi.core.models'`

**Step 3: Write minimal implementation**

`src/rsi/core/models.py`:
```python
"""Core data models for reddit-stash content."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class Post:
    """A reddit-stash saved post."""

    id: str
    subreddit: str
    title: str
    body: str
    permalink: str
    file_path: str
    author: Optional[str] = None
    timestamp: Optional[datetime] = None
    flair: Optional[str] = None
    score: int = 0
    comment_count: int = 0

    def search_text(self) -> str:
        """Text representation for embedding/search. Combines title + body."""
        return f"{self.title}\n\n{self.body}"


@dataclass(frozen=True)
class Comment:
    """A reddit-stash saved comment."""

    id: str
    subreddit: str
    author: str
    body: str
    permalink: str
    file_path: str
    score: int = 0
    timestamp: Optional[datetime] = None
    parent_title: Optional[str] = None
    parent_id: Optional[str] = None

    def search_text(self) -> str:
        """Text representation for embedding/search. Includes parent title for context."""
        parts = []
        if self.parent_title:
            parts.append(self.parent_title)
        parts.append(self.body)
        return "\n\n".join(parts)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_models -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/rsi/core/models.py tests/test_models.py
git commit -m "feat: add Post and Comment data models with search_text()"
```

---

## Task 3: Markdown Parser — Posts

**Files:**
- Create: `src/rsi/core/parser.py`
- Create: `tests/test_parser.py`
- Create: `tests/fixtures/` (test data)

**Step 1: Create test fixtures**

Create `tests/fixtures/sample_post.md`:
```markdown
---
id: 1grc5bi
subreddit: /r/Python
timestamp: 2024-11-14 18:52:30
author: /u/complexrexton
flair: Showcase
comments: 7
permalink: https://reddit.com/r/Python/comments/1grc5bi/squeakycleantext_a_modular_text_processing/
---

# SqueakyCleanText: A Modular Text Processing Library

**Upvotes:** 12 | **Permalink:** [Link](https://reddit.com/...)

GitHub: [SqueakyCleanText](https://github.com/rhnfzl/SqueakyCleanText)

Happy to share **SqueakyCleanText**, a Python library for text preprocessing.
```

Create `tests/fixtures/sample_comment.md`:
```markdown
---
Comment by /u/complexrexton
- **Upvotes:** 1 | **Permalink:** [Link](https://reddit.com/r/fitbod/comments/1njd7m5/fitbod_screen_stuck/nepvcs7/)
Thank you it's working now

---

## Context: Parent Comment by /u/JessAtFitbod
- **Upvotes:** 1 | **Permalink:** [Link](https://reddit.com/r/fitbod/comments/1njd7m5/fitbod_screen_stuck/nepk9wc/)
We just released a fix!

---

## Context: Post by /u/complexrexton
- **Title:** Fitbod Screen Stuck
- **Upvotes:** 1 | **Permalink:** [Link](https://reddit.com/r/fitbod/comments/1njd7m5/fitbod_screen_stuck/)
```

**Step 2: Write the failing test**

`tests/test_parser.py`:
```python
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
```

**Step 3: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_parser -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rsi.core.parser'`

**Step 4: Write minimal implementation**

`src/rsi/core/parser.py`:
```python
"""Parse reddit-stash markdown files into Post/Comment data models."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from rsi.core.models import Comment, Post


def parse_post(file_path: Path) -> Post:
    """Parse a reddit-stash POST_*.md file into a Post dataclass.

    Posts have YAML frontmatter between --- delimiters with fields:
    id, subreddit, timestamp, author, flair, comments, permalink.
    Body follows the closing --- delimiter.
    """
    content = file_path.read_text(encoding="utf-8")
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"No YAML frontmatter found in {file_path}")

    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].strip()

    # Extract score from "**Upvotes:** N" pattern in body
    score = 0
    score_match = re.search(r"\*\*Upvotes:\*\*\s*(\d+)", body)
    if score_match:
        score = int(score_match.group(1))

    # Strip /r/ and /u/ prefixes
    subreddit = _strip_prefix(frontmatter.get("subreddit", ""), "/r/")
    author = _strip_prefix(frontmatter.get("author", ""), "/u/")

    # Parse timestamp
    timestamp = _parse_timestamp(frontmatter.get("timestamp"))

    return Post(
        id=str(frontmatter["id"]),
        subreddit=subreddit,
        title=_extract_title(body),
        body=body,
        permalink=frontmatter.get("permalink", ""),
        file_path=str(file_path),
        author=author or None,
        timestamp=timestamp,
        flair=frontmatter.get("flair"),
        score=score,
        comment_count=int(frontmatter.get("comments", 0)),
    )


def parse_comment(file_path: Path, subreddit: str) -> Comment:
    """Parse a reddit-stash COMMENT_*.md file into a Comment dataclass.

    Comments do NOT have YAML frontmatter. They have inline markdown:
    - First line: "---"
    - "Comment by /u/Author"
    - "- **Upvotes:** N | **Permalink:** [Link](url)"
    - Body text follows
    """
    content = file_path.read_text(encoding="utf-8")

    # Extract author from "Comment by /u/author" pattern
    author = ""
    author_match = re.search(r"Comment by /u/(\w+)", content)
    if author_match:
        author = author_match.group(1)

    # Extract comment body — text between first "---" block and next "---"
    # The comment's own text is between the first permalink line and next ---
    body = _extract_comment_body(content)

    # Extract permalink and derive comment ID from it
    comment_id = ""
    permalink = ""
    permalink_match = re.search(
        r"\*\*Permalink:\*\*\s*\[Link\]\((https://reddit\.com/[^)]+/(\w+)/)\)",
        content,
    )
    if permalink_match:
        permalink = permalink_match.group(1)
        comment_id = permalink_match.group(2)

    # Extract score
    score = 0
    score_match = re.search(r"\*\*Upvotes:\*\*\s*(\d+)", content)
    if score_match:
        score = int(score_match.group(1))

    # Extract parent title from "- **Title:** ..."
    parent_title = None
    title_match = re.search(r"\*\*Title:\*\*\s*(.+)", content)
    if title_match:
        parent_title = title_match.group(1).strip()

    # Derive ID from filename if not found in permalink
    if not comment_id:
        fname = file_path.stem  # COMMENT_nepvcs7 -> COMMENT_nepvcs7
        if fname.startswith("COMMENT_"):
            comment_id = fname[8:]

    return Comment(
        id=comment_id,
        subreddit=subreddit,
        author=author,
        body=body,
        permalink=permalink,
        file_path=str(file_path),
        score=score,
        parent_title=parent_title,
    )


def _extract_title(body: str) -> str:
    """Extract the markdown H1 title from body text."""
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _extract_comment_body(content: str) -> str:
    """Extract the comment's own text from the markdown.

    The comment body is the text after the permalink line
    and before the next '---' separator.
    """
    lines = content.splitlines()
    body_lines = []
    in_body = False
    for line in lines:
        if in_body:
            if line.strip() == "---":
                break
            body_lines.append(line)
        elif "**Permalink:**" in line:
            in_body = True
    return "\n".join(body_lines).strip()


def _strip_prefix(value: str, prefix: str) -> str:
    """Remove prefix like /r/ or /u/ from a string."""
    value = str(value).strip()
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def _parse_timestamp(value) -> Optional[datetime]:
    """Parse a timestamp string into a datetime object."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return None
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_parser -v`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add src/rsi/core/parser.py tests/test_parser.py tests/fixtures/
git commit -m "feat: add markdown parser for posts and comments"
```

---

## Task 4: file_log.json Reader & Directory Scanner

**Files:**
- Create: `src/rsi/core/file_log.py`
- Create: `src/rsi/core/scanner.py`
- Create: `tests/test_scanner.py`
- Create: `tests/fixtures/sample_file_log.json`

**Step 1: Create test fixture**

`tests/fixtures/sample_file_log.json`:
```json
{
    "1grc5bi-Python-Submission-POST": {
        "subreddit": "Python",
        "type": "Submission",
        "file_path": "Python/POST_1grc5bi.md"
    },
    "nepvcs7-fitbod-Comment-COMMENT": {
        "subreddit": "fitbod",
        "type": "Comment",
        "file_path": "fitbod/COMMENT_nepvcs7.md"
    }
}
```

**Step 2: Write the failing test**

`tests/test_scanner.py`:
```python
"""Tests for file_log reader and directory scanner."""
import unittest
import json
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
            # Create a subreddit folder with a post
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
```

**Step 3: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_scanner -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write implementations**

`src/rsi/core/file_log.py`:
```python
"""Read and filter reddit-stash file_log.json entries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


FileLogEntry = Dict[str, str]
FileLog = Dict[str, FileLogEntry]


def read_file_log(path: Path) -> FileLog:
    """Read file_log.json and return all entries."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_post_entries(log: FileLog) -> FileLog:
    """Filter file_log entries to only Submission (post) entries."""
    return {k: v for k, v in log.items() if v.get("type") == "Submission"}


def get_comment_entries(log: FileLog) -> FileLog:
    """Filter file_log entries to only Comment entries."""
    return {k: v for k, v in log.items() if v.get("type") == "Comment"}
```

`src/rsi/core/scanner.py`:
```python
"""Scan a reddit-stash output directory and parse all markdown files."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from rsi.core.models import Comment, Post
from rsi.core.parser import parse_comment, parse_post

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a reddit-stash output directory."""

    posts: List[Post] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def scan_directory(reddit_dir: Path) -> ScanResult:
    """Scan a reddit-stash output directory for all POST_*.md and COMMENT_*.md files.

    Expects structure: reddit_dir/{subreddit}/POST_*.md and COMMENT_*.md
    """
    result = ScanResult()

    if not reddit_dir.is_dir():
        logger.warning("Directory does not exist: %s", reddit_dir)
        return result

    for subreddit_dir in sorted(reddit_dir.iterdir()):
        if not subreddit_dir.is_dir():
            continue

        subreddit_name = subreddit_dir.name

        for md_file in sorted(subreddit_dir.glob("*.md")):
            try:
                if md_file.name.startswith("POST_"):
                    post = parse_post(md_file)
                    result.posts.append(post)
                elif md_file.name.startswith("COMMENT_"):
                    comment = parse_comment(md_file, subreddit=subreddit_name)
                    result.comments.append(comment)
            except Exception as e:
                error_msg = f"Error parsing {md_file}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)

    return result
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_scanner -v`
Expected: All 6 tests PASS

**Step 6: Commit**

```bash
git add src/rsi/core/file_log.py src/rsi/core/scanner.py tests/test_scanner.py tests/fixtures/sample_file_log.json
git commit -m "feat: add file_log reader and directory scanner"
```

---

## Task 5: Typer CLI Skeleton

**Files:**
- Create: `src/rsi/cli.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

`tests/test_cli.py`:
```python
"""Tests for CLI commands."""
import unittest
import tempfile
from pathlib import Path
from typer.testing import CliRunner


class TestCLI(unittest.TestCase):
    def test_version_flag(self):
        from rsi.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("0.1.0", result.output)

    def test_scan_command_on_empty_dir(self):
        from rsi.cli import app

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["scan", tmpdir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("0 posts", result.output)
            self.assertIn("0 comments", result.output)

    def test_scan_command_on_nonexistent_dir(self):
        from rsi.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["scan", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_cli -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

`src/rsi/cli.py`:
```python
"""CLI entry point for reddit-stash-insights."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from rsi import __version__

app = typer.Typer(
    name="rsi",
    help="Semantic search, knowledge graph, and AI chat for reddit-stash archives.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"reddit-stash-insights {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True),
):
    """reddit-stash-insights: search, graph, analytics, and chat for your Reddit archive."""
    pass


@app.command()
def scan(
    reddit_dir: Path = typer.Argument(..., help="Path to reddit-stash output directory", exists=True),
):
    """Scan a reddit-stash output directory and report contents."""
    from rsi.core.scanner import scan_directory

    result = scan_directory(reddit_dir)
    typer.echo(f"Found {len(result.posts)} posts and {len(result.comments)} comments")
    if result.errors:
        typer.echo(f"  ({len(result.errors)} parse errors)", err=True)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_cli -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/rsi/cli.py tests/test_cli.py
git commit -m "feat: add Typer CLI with scan command and --version flag"
```

---

## Task 6: BGE-M3 Embedder

**Files:**
- Create: `src/rsi/indexer/__init__.py`
- Create: `src/rsi/indexer/embedder.py`
- Create: `tests/test_embedder.py`

**Prerequisite:** `pip install FlagEmbedding sentence-transformers` (the `[search]` dep group)

**Step 1: Write the failing test**

`tests/test_embedder.py`:
```python
"""Tests for BGE-M3 embedder.

NOTE: These tests download the BGE-M3 model (~2GB) on first run.
Skip with: PYTHONPATH=src python -m unittest tests.test_embedder -v -k "not slow"
"""
import unittest
import os


SKIP_SLOW = os.environ.get("RSI_SKIP_SLOW_TESTS", "0") == "1"


class TestEmbedder(unittest.TestCase):
    @unittest.skipIf(SKIP_SLOW, "Skipping slow model download test")
    def test_embed_returns_dense_and_sparse(self):
        from rsi.indexer.embedder import Embedder

        embedder = Embedder()
        result = embedder.embed(["Hello world"])
        self.assertIn("dense", result)
        self.assertIn("sparse", result)
        self.assertEqual(len(result["dense"]), 1)
        self.assertEqual(len(result["dense"][0]), 1024)  # BGE-M3 dimensions

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model download test")
    def test_embed_batch(self):
        from rsi.indexer.embedder import Embedder

        embedder = Embedder()
        texts = ["First text", "Second text", "Third text"]
        result = embedder.embed(texts)
        self.assertEqual(len(result["dense"]), 3)
        self.assertEqual(len(result["sparse"]), 3)

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model download test")
    def test_embed_query(self):
        from rsi.indexer.embedder import Embedder

        embedder = Embedder()
        result = embedder.embed_query("search query")
        self.assertIn("dense", result)
        self.assertEqual(len(result["dense"]), 1024)


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_embedder -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write implementation**

`src/rsi/indexer/__init__.py`:
```python
"""Search indexer — BGE-M3 embeddings + LanceDB vector store."""
```

`src/rsi/indexer/embedder.py`:
```python
"""BGE-M3 embedding model — produces dense + sparse vectors in one call."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Model name constant
BGE_M3_MODEL = "BAAI/bge-m3"


class Embedder:
    """Wrapper around BGE-M3 for generating dense and sparse embeddings."""

    def __init__(self, model_name: str = BGE_M3_MODEL, use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading embedding model: %s", model_name)
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        logger.info("Model loaded successfully")

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Embed a batch of texts, returning dense and sparse vectors.

        Returns:
            {"dense": list of 1024-dim vectors, "sparse": list of sparse weight dicts}
        """
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": output["dense_vecs"].tolist(),
            "sparse": output["lexical_weights"],
        }

    def embed_query(self, query: str) -> Dict[str, Any]:
        """Embed a single query for search.

        Returns:
            {"dense": 1024-dim vector, "sparse": sparse weight dict}
        """
        result = self.embed([query])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_embedder -v`
Expected: All 3 tests PASS (first run downloads ~2GB model, takes a few minutes)

**Step 5: Commit**

```bash
git add src/rsi/indexer/ tests/test_embedder.py
git commit -m "feat: add BGE-M3 embedder with dense + sparse vector generation"
```

---

## Task 7: LanceDB Vector Store

**Files:**
- Create: `src/rsi/indexer/vector_store.py`
- Create: `tests/test_vector_store.py`

**Prerequisite:** `pip install lancedb`

**Step 1: Write the failing test**

`tests/test_vector_store.py`:
```python
"""Tests for LanceDB vector store."""
import unittest
import tempfile
from pathlib import Path


class TestVectorStore(unittest.TestCase):
    def test_create_and_query_table(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")

            # Add sample records with fake 1024-dim vectors
            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust ownership and borrowing explained",
                    "vector": [0.2] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            self.assertEqual(store.count(), 2)

    def test_search_returns_results(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")

            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)

            results = store.vector_search([0.1] * 1024, limit=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["id"], "abc")

    def test_fts_search(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")

            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python async programming tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust ownership explained",
                    "vector": [0.2] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)
            store.create_fts_index()

            results = store.fts_search("Python async", limit=5)
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["id"], "abc")

    def test_subreddit_filter(self):
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=Path(tmpdir) / "test.lance")

            records = [
                {
                    "id": "abc",
                    "subreddit": "Python",
                    "text": "Python tutorial",
                    "vector": [0.1] * 1024,
                    "score": 10,
                    "timestamp": "2024-01-01",
                    "file_path": "Python/POST_abc.md",
                    "content_type": "post",
                },
                {
                    "id": "def",
                    "subreddit": "rust",
                    "text": "Rust tutorial",
                    "vector": [0.1] * 1024,
                    "score": 25,
                    "timestamp": "2024-02-01",
                    "file_path": "rust/POST_def.md",
                    "content_type": "post",
                },
            ]
            store.add_records(records)

            results = store.vector_search([0.1] * 1024, limit=10, subreddit="Python")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["subreddit"], "Python")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_vector_store -v`
Expected: FAIL

**Step 3: Write implementation**

`src/rsi/indexer/vector_store.py`:
```python
"""LanceDB vector store for reddit-stash content."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import lancedb

logger = logging.getLogger(__name__)

TABLE_NAME = "posts"


class VectorStore:
    """LanceDB-backed vector store with FTS (BM25) support."""

    def __init__(self, db_path: Path):
        self._db = lancedb.connect(str(db_path))
        self._table = None

    def add_records(self, records: List[Dict[str, Any]]) -> None:
        """Add records to the vector store. Creates table on first call, appends after."""
        if self._table is None:
            try:
                self._table = self._db.open_table(TABLE_NAME)
                self._table.add(records)
            except Exception:
                self._table = self._db.create_table(TABLE_NAME, data=records)
        else:
            self._table.add(records)
        logger.info("Added %d records to vector store", len(records))

    def create_fts_index(self) -> None:
        """Create a Tantivy full-text search index on the 'text' column."""
        if self._table is None:
            raise RuntimeError("No table exists yet. Add records first.")
        self._table.create_fts_index("text", replace=True)
        logger.info("Created FTS index on 'text' column")

    def vector_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Pure vector similarity search."""
        if self._table is None:
            return []
        q = self._table.search(query_vector).limit(limit)
        if subreddit:
            q = q.where(f"subreddit = '{subreddit}'")
        return q.to_list()

    def fts_search(
        self,
        query: str,
        limit: int = 10,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text (BM25) keyword search via Tantivy."""
        if self._table is None:
            return []
        q = self._table.search(query, query_type="fts").limit(limit)
        if subreddit:
            q = q.where(f"subreddit = '{subreddit}'")
        return q.to_list()

    def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        limit: int = 10,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining BM25 + vector similarity with RRF reranking."""
        from lancedb.rerankers import RRFReranker

        if self._table is None:
            return []
        reranker = RRFReranker(k=60)
        q = (
            self._table.search(query, query_type="hybrid", vector=query_vector)
            .rerank(reranker=reranker)
            .limit(limit)
        )
        if subreddit:
            q = q.where(f"subreddit = '{subreddit}'")
        return q.to_list()

    def count(self) -> int:
        """Return the number of records in the store."""
        if self._table is None:
            return 0
        return self._table.count_rows()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_vector_store -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/rsi/indexer/vector_store.py tests/test_vector_store.py
git commit -m "feat: add LanceDB vector store with FTS, vector, and hybrid search"
```

---

## Task 8: Search Orchestrator

**Files:**
- Create: `src/rsi/indexer/search.py`
- Create: `tests/test_search.py`

This module ties together the embedder + vector store into a high-level search interface.

**Step 1: Write the failing test**

`tests/test_search.py`:
```python
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

            # Index some content
            engine.index_texts(
                texts=["Python async await tutorial", "Rust memory safety"],
                ids=["abc", "def"],
                metadata=[
                    {"subreddit": "Python", "score": 10, "timestamp": "2024-01-01",
                     "file_path": "Python/POST_abc.md", "content_type": "post"},
                    {"subreddit": "rust", "score": 25, "timestamp": "2024-02-01",
                     "file_path": "rust/POST_def.md", "content_type": "post"},
                ],
            )

            results = engine.search("Python async", limit=2)
            self.assertGreater(len(results), 0)


class TestSearchMode(unittest.TestCase):
    def test_search_mode_enum(self):
        from rsi.indexer.search import SearchMode

        self.assertEqual(SearchMode.HYBRID.value, "hybrid")
        self.assertEqual(SearchMode.SEMANTIC.value, "semantic")
        self.assertEqual(SearchMode.KEYWORD.value, "keyword")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_search -v`
Expected: FAIL

**Step 3: Write implementation**

`src/rsi/indexer/search.py`:
```python
"""High-level search engine orchestrating embedder + vector store."""
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from rsi.indexer.embedder import Embedder
from rsi.indexer.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"


class SearchEngine:
    """Orchestrates BGE-M3 embedding and LanceDB search."""

    def __init__(self, db_path: Path, embedder: Optional[Embedder] = None):
        self._store = VectorStore(db_path=db_path)
        self._embedder = embedder

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def index_texts(
        self,
        texts: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Embed texts and store them in the vector database."""
        embedder = self._get_embedder()
        embeddings = embedder.embed(texts)

        records = []
        for i, text in enumerate(texts):
            record = {
                "id": ids[i],
                "text": text,
                "vector": embeddings["dense"][i],
                **metadata[i],
            }
            records.append(record)

        self._store.add_records(records)
        self._store.create_fts_index()
        logger.info("Indexed %d texts", len(texts))

    def search(
        self,
        query: str,
        limit: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        subreddit: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search indexed content."""
        if mode == SearchMode.KEYWORD:
            return self._store.fts_search(query, limit=limit, subreddit=subreddit)

        embedder = self._get_embedder()
        query_emb = embedder.embed_query(query)

        if mode == SearchMode.SEMANTIC:
            return self._store.vector_search(
                query_emb["dense"], limit=limit, subreddit=subreddit
            )

        # Default: hybrid
        return self._store.hybrid_search(
            query=query,
            query_vector=query_emb["dense"],
            limit=limit,
            subreddit=subreddit,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_search -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/rsi/indexer/search.py tests/test_search.py
git commit -m "feat: add search engine with hybrid/semantic/keyword modes"
```

---

## Task 9: CLI `index` and `search` Commands

**Files:**
- Modify: `src/rsi/cli.py`
- Create: `src/rsi/config.py`
- Create: `tests/test_cli_search.py`

**Step 1: Write the failing test**

`tests/test_cli_search.py`:
```python
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
            # Create minimal reddit-stash structure
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
            # Should handle gracefully (0 results or error message)
            self.assertIn("0 results", result.output.lower())


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_cli_search -v`
Expected: FAIL

**Step 3: Write implementation**

`src/rsi/config.py`:
```python
"""Configuration defaults for reddit-stash-insights."""
from __future__ import annotations

from pathlib import Path

DEFAULT_DB_DIR = Path.home() / ".rsi" / "index"
DEFAULT_DB_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_PATH = DEFAULT_DB_DIR / "vectors.lance"
```

Update `src/rsi/cli.py` — add `index` and `search` commands:
```python
"""CLI entry point for reddit-stash-insights."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from rsi import __version__

app = typer.Typer(
    name="rsi",
    help="Semantic search, knowledge graph, and AI chat for reddit-stash archives.",
    no_args_is_help=True,
)


def version_callback(value: bool):
    if value:
        typer.echo(f"reddit-stash-insights {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(None, "--version", callback=version_callback, is_eager=True),
):
    """reddit-stash-insights: search, graph, analytics, and chat for your Reddit archive."""
    pass


@app.command()
def scan(
    reddit_dir: Path = typer.Argument(..., help="Path to reddit-stash output directory", exists=True),
):
    """Scan a reddit-stash output directory and report contents."""
    from rsi.core.scanner import scan_directory

    result = scan_directory(reddit_dir)
    typer.echo(f"Found {len(result.posts)} posts and {len(result.comments)} comments")
    if result.errors:
        typer.echo(f"  ({len(result.errors)} parse errors)", err=True)


@app.command()
def index(
    reddit_dir: Path = typer.Argument(..., help="Path to reddit-stash output directory", exists=True),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path for vector database"),
):
    """Build or update the vector search index from reddit-stash content."""
    from rsi.config import DEFAULT_DB_PATH
    from rsi.core.scanner import scan_directory
    from rsi.indexer.search import SearchEngine

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    typer.echo(f"Scanning {reddit_dir}...")
    scan_result = scan_directory(reddit_dir)
    typer.echo(f"Found {len(scan_result.posts)} posts, {len(scan_result.comments)} comments")

    if not scan_result.posts and not scan_result.comments:
        typer.echo("Nothing to index.")
        raise typer.Exit()

    typer.echo("Loading embedding model (first run downloads ~2GB)...")
    engine = SearchEngine(db_path=db_path)

    # Prepare texts and metadata
    texts = []
    ids = []
    metadata = []

    for post in scan_result.posts:
        texts.append(post.search_text())
        ids.append(post.id)
        metadata.append({
            "subreddit": post.subreddit,
            "score": post.score,
            "timestamp": str(post.timestamp or ""),
            "file_path": post.file_path,
            "content_type": "post",
        })

    for comment in scan_result.comments:
        texts.append(comment.search_text())
        ids.append(comment.id)
        metadata.append({
            "subreddit": comment.subreddit,
            "score": comment.score,
            "timestamp": str(comment.timestamp or ""),
            "file_path": comment.file_path,
            "content_type": "comment",
        })

    typer.echo(f"Embedding {len(texts)} items...")
    engine.index_texts(texts=texts, ids=ids, metadata=metadata)
    typer.echo(f"Indexed {len(texts)} items to {db_path}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    db_path: Optional[Path] = typer.Option(None, "--db-path", help="Path to vector database"),
    subreddit: Optional[str] = typer.Option(None, "--subreddit", "-s", help="Filter by subreddit"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, semantic, keyword"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results"),
):
    """Search your indexed reddit-stash content."""
    from rsi.config import DEFAULT_DB_PATH
    from rsi.indexer.search import SearchEngine, SearchMode

    if db_path is None:
        db_path = DEFAULT_DB_PATH

    if not db_path.exists():
        typer.echo("No index found. Run 'rsi index' first. 0 results found.")
        raise typer.Exit()

    search_mode = SearchMode(mode)
    engine = SearchEngine(db_path=db_path)
    results = engine.search(query, limit=limit, mode=search_mode, subreddit=subreddit)

    if not results:
        typer.echo(f"0 results found for: {query}")
        return

    typer.echo(f"{len(results)} results for: {query}\n")
    for i, r in enumerate(results, 1):
        sub = r.get("subreddit", "?")
        score = r.get("score", 0)
        text_preview = r.get("text", "")[:120].replace("\n", " ")
        typer.echo(f"  {i}. [r/{sub}] (score: {score}) {text_preview}")
        typer.echo(f"     File: {r.get('file_path', '?')}")
        typer.echo()
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_cli_search -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/rsi/cli.py src/rsi/config.py tests/test_cli_search.py
git commit -m "feat: add CLI index and search commands with hybrid search"
```

---

## Task 10: Integration Test with Real reddit-stash Data

**Files:**
- Create: `tests/test_integration.py`

This task runs the full pipeline against the actual reddit-stash output directory at `../reddit-stash/reddit/`.

**Step 1: Write the integration test**

`tests/test_integration.py`:
```python
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
        # Verify no critical parse errors
        self.assertLess(
            len(result.errors), len(result.posts),
            "More errors than posts — parser likely broken"
        )

    def test_full_index_and_search(self):
        from rsi.core.scanner import scan_directory
        from rsi.indexer.search import SearchEngine

        scan_result = scan_directory(REDDIT_DIR)
        self.assertGreater(len(scan_result.posts), 0)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.lance"
            engine = SearchEngine(db_path=db_path)

            # Index first 5 posts (keep test fast)
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

            # Search for something that should match
            first_title = posts[0].title
            if first_title:
                results = engine.search(first_title, limit=3)
                self.assertGreater(len(results), 0, f"Should find results for: {first_title}")


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run the integration test**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest tests.test_integration -v`
Expected: All tests PASS (may take 1-2 minutes for model loading + embedding)

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test against real reddit-stash data"
```

---

## Task 11: Lint and Final Cleanup

**Step 1: Run ruff**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && ruff check src/ tests/`

**Step 2: Fix any lint issues**

Run: `ruff check --fix src/ tests/`

**Step 3: Run full test suite**

Run: `cd /Users/rehan-8v/8vance/portfolio/reddit-stash-insights && PYTHONPATH=src python -m unittest discover tests -v`
Expected: All tests PASS

**Step 4: Commit if fixes were made**

```bash
git add -A && git commit -m "chore: lint fixes"
```

---

## Summary

After completing all 11 tasks, you'll have:

1. **Project scaffold** — `pyproject.toml` with modular dep groups, proper package structure
2. **Data models** — `Post` and `Comment` frozen dataclasses with `search_text()`
3. **Parser** — Handles both YAML-frontmatter posts and inline-markdown comments
4. **Scanner** — Walks reddit-stash directory tree, parses all files
5. **Embedder** — BGE-M3 dense + sparse vector generation
6. **Vector store** — LanceDB with Tantivy FTS, vector search, and hybrid search (RRF)
7. **Search engine** — Orchestrates embedder + store, supports 3 search modes
8. **CLI** — `rsi scan`, `rsi index`, `rsi search` commands
9. **Tests** — Unit + integration covering all components
10. **Linted** — ruff-clean codebase

**Next phase:** Phase 2 (Knowledge Graph) builds on this foundation — uses the same parser/scanner and embedding vectors for topic extraction and graph construction.
