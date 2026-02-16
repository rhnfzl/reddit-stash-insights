"""Core data models for reddit-stash content."""
from __future__ import annotations

from dataclasses import dataclass
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
