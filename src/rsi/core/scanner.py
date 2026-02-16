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
    """Scan a reddit-stash output directory for all POST_*.md and COMMENT_*.md files."""
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
