"""Parse reddit-stash markdown files into Post/Comment data models."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from rsi.core.models import Comment, Post

# reddit-stash writes {TYPE}_{id}.md; TYPE is one of these (utils/path_security.py there).
POST_TYPES = {"POST", "SAVED_POST", "UPVOTE_POST", "GDPR_POST"}
COMMENT_TYPES = {"COMMENT", "SAVED_COMMENT", "UPVOTE_COMMENT", "GDPR_COMMENT"}
_SOURCES = {"SAVED": "saved", "UPVOTE": "upvoted", "GDPR": "gdpr"}


_TYPE_RE = re.compile(
    r"^(SAVED_POST|SAVED_COMMENT|UPVOTE_POST|UPVOTE_COMMENT|GDPR_POST|GDPR_COMMENT|POST|COMMENT)_(.+)$"
)
_FRONTMATTER_RE = re.compile(r"\A---[ \t]*\n(.*?)\n---[ \t]*(?:\n|\Z)(.*)\Z", re.DOTALL)


def item_type(file_path: Path) -> str:
    """Return the reddit-stash type token of a file name, e.g. 'SAVED_POST' ('' if unknown).

    Matches the known prefixes, so an id that itself contains '_' (t3_abc) still works."""
    m = _TYPE_RE.match(file_path.stem)
    return m.group(1) if m else ""


def item_id(file_path: Path) -> str:
    """The id part of a reddit-stash file name."""
    m = _TYPE_RE.match(file_path.stem)
    return m.group(2) if m else file_path.stem


def split_frontmatter(content: str) -> tuple[str, str] | None:
    """(frontmatter, body) when the file starts with a --- block; delimiters must be whole lines,
    so a '---' inside a value or the body does not end the block."""
    m = _FRONTMATTER_RE.match(content)
    return (m.group(1), m.group(2)) if m else None


def item_source(file_path: Path) -> str:
    """Where the item came from: own, saved, upvoted or gdpr."""
    return _SOURCES.get(item_type(file_path).split("_")[0], "own")


def folder_subreddit(file_path: Path) -> str:
    """Subreddit from the parent folder; CSV-only GDPR exports use an 'r_' prefix there."""
    name = file_path.parent.name
    return name.removeprefix("r_") if file_path.name.startswith("GDPR_") else name


def parse_post(file_path: Path) -> Post:
    """Parse a reddit-stash post file (POST_, SAVED_POST_, UPVOTE_POST_, GDPR_POST_).

    Posts have YAML frontmatter between --- delimiters with fields:
    id, subreddit, timestamp, author, flair, comments, permalink.
    Body follows the closing --- delimiter.
    """
    content = file_path.read_text(encoding="utf-8")
    parts = split_frontmatter(content)
    if parts is None:
        raise ValueError(f"No YAML frontmatter found in {file_path}")

    try:
        frontmatter = yaml.safe_load(parts[0])
    except yaml.YAMLError:
        # Older reddit-stash wrote unquoted values (e.g. "flair: Use: X"); read them line by line
        frontmatter = dict(
            line.split(": ", 1) for line in parts[0].strip().splitlines() if ": " in line
        )
    body = parts[1].strip()

    # Extract score from "**Upvotes:** N" pattern in body
    score = 0
    score_match = re.search(r"\*\*Upvotes:\*\*\s*(\d+)", body)
    if score_match:
        score = int(score_match.group(1))

    # Strip /r/ and /u/ prefixes; CSV-only GDPR exports have no subreddit field
    subreddit = _strip_prefix(frontmatter.get("subreddit", ""), "/r/") or folder_subreddit(file_path)
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
        source=item_source(file_path),
    )


def parse_comment(file_path: Path, subreddit: str) -> Comment:
    """Parse a reddit-stash comment file (COMMENT_, SAVED_COMMENT_, UPVOTE_COMMENT_, GDPR_COMMENT_).

    Newer reddit-stash versions (and GDPR CSV-only exports) write YAML frontmatter;
    see _parse_frontmatter_comment. Older comment files have inline markdown only:
    - First line: "---"
    - "Comment by /u/Author"
    - "- **Upvotes:** N | **Permalink:** [Link](url)"
    - Body text follows
    """
    content = file_path.read_text(encoding="utf-8")
    frontmatter = _frontmatter(content)
    if frontmatter is not None:
        return _parse_frontmatter_comment(file_path, subreddit, content, frontmatter)

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
        comment_id = item_id(file_path)

    return Comment(
        id=comment_id,
        subreddit=subreddit,
        author=author,
        body=body,
        permalink=permalink,
        file_path=str(file_path),
        score=score,
        parent_title=parent_title,
        source=item_source(file_path),
    )


def _frontmatter(content: str) -> dict | None:
    """Return the YAML frontmatter dict if the file starts with one that has an id."""
    parts = split_frontmatter(content)
    if parts is None:
        return None
    try:
        data = yaml.safe_load(parts[0])
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) and "id" in data else None


def _parse_frontmatter_comment(file_path: Path, subreddit: str, content: str, fm: dict) -> Comment:
    """Comment with frontmatter: id, author, body, score, permalink (GDPR CSV-only: id, title, permalink)."""
    # CSV-only GDPR files have no body field; the markdown after the frontmatter is all there is
    body = str(fm.get("body") or "").strip() or split_frontmatter(content)[1].strip()
    title_match = re.search(r"\*\*Title:\*\*\s*(.+)", content)
    try:
        score = int(fm.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    return Comment(
        id=str(fm["id"]),
        subreddit=subreddit,
        author=_strip_prefix(fm.get("author") or "", "/u/"),
        body=body,
        permalink=str(fm.get("permalink") or ""),
        file_path=str(file_path),
        score=score,
        parent_title=title_match.group(1).strip() if title_match else None,
        source=item_source(file_path),
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
