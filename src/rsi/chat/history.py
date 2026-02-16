"""Conversation history manager for multi-turn RAG chat."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rsi.chat.providers.base import ChatMessage


@dataclass
class ConversationTurn:
    """A single Q&A turn in the conversation."""

    query: str
    response: str
    sources: list[dict]
    timestamp: str  # ISO format string


class ChatHistory:
    """Manages conversation history with eviction and persistence."""

    def __init__(self, max_turns: int = 10):
        self._turns: list[ConversationTurn] = []
        self._max_turns = max_turns

    def add_turn(self, query: str, response: str, sources: list[dict]) -> None:
        self._turns.append(
            ConversationTurn(
                query=query,
                response=response,
                sources=sources,
                timestamp=datetime.now().isoformat(),
            )
        )
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns :]

    def to_messages(self) -> list[ChatMessage]:
        messages = []
        for turn in self._turns:
            messages.append(ChatMessage(role="user", content=turn.query))
            messages.append(ChatMessage(role="assistant", content=turn.response))
        return messages

    def last_sources(self) -> list[dict]:
        if not self._turns:
            return []
        return self._turns[-1].sources

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(t) for t in self._turns], f, indent=2)

    @classmethod
    def load(cls, path: Path, max_turns: int = 10) -> ChatHistory:
        history = cls(max_turns=max_turns)
        if path.exists():
            with open(path) as f:
                for item in json.load(f):
                    history._turns.append(ConversationTurn(**item))
        return history
