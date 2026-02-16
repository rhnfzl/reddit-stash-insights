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

    # Extract comment body — text between first permalink line and next ---
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
        fname = file_path.stem
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
