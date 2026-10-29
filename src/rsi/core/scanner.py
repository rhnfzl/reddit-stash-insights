"""Scan a reddit-stash output directory and parse all markdown files."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from rsi.core.models import Comment, Post
from rsi.core.parser import (
    COMMENT_TYPES,
    POST_TYPES,
    folder_subreddit,
    item_source,
    item_type,
    parse_comment,
    parse_post,
)

logger = logging.getLogger(__name__)

_SOURCE_RANK = {"own": 0, "saved": 1, "upvoted": 2, "gdpr": 3}


@dataclass
class ScanResult:
    """Result of scanning a reddit-stash output directory."""

    posts: List[Post] = field(default_factory=list)
    comments: List[Comment] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def scan_directory(reddit_dir: Path) -> ScanResult:
    """Scan a reddit-stash output directory for every post and comment file type
    (own, SAVED_, UPVOTE_ and GDPR_ prefixes)."""
    result = ScanResult()

    if not reddit_dir.is_dir():
        logger.warning("Directory does not exist: %s", reddit_dir)
        return result

    # The same item can exist as several files (own, saved, upvoted, GDPR placeholder).
    # Parse the fullest first, so own > saved > upvoted > gdpr wins the duplicate check.
    files = [
        md_file
        for subreddit_dir in reddit_dir.iterdir()
        if subreddit_dir.is_dir()
        for md_file in subreddit_dir.glob("*.md")
    ]
    files.sort(key=lambda f: (_SOURCE_RANK.get(item_source(f), len(_SOURCE_RANK)), str(f)))
    seen = set()

    for md_file in files:
        kind = item_type(md_file)
        try:
            if kind in POST_TYPES:
                post = parse_post(md_file)
                if ("post", post.id) not in seen:
                    seen.add(("post", post.id))
                    result.posts.append(post)
            elif kind in COMMENT_TYPES:
                comment = parse_comment(md_file, subreddit=folder_subreddit(md_file))
                if ("comment", comment.id) not in seen:
                    seen.add(("comment", comment.id))
                    result.comments.append(comment)
        except Exception as e:
            error_msg = f"Error parsing {md_file}: {e}"
            logger.warning(error_msg)
            result.errors.append(error_msg)

    return result
