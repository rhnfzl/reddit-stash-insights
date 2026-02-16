"""Direct RAG engine -- orchestrates SearchEngine + prompt + LLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from rsi.chat.history import ChatHistory
from rsi.chat.prompt import build_messages
from rsi.chat.providers.base import LLMProvider
from rsi.indexer.search import SearchEngine, SearchMode


@dataclass
class ChatResponse:
    """Result from a RAG chat query."""

    answer: str
    sources: list[dict]


class DirectEngine:
    """Direct RAG: our SearchEngine retrieves context, LLM generates answer."""

    def __init__(
        self,
        search_engine: SearchEngine,
        llm: LLMProvider,
        search_mode: SearchMode = SearchMode.HYBRID,
        max_history_turns: int = 10,
    ):
        self._search = search_engine
        self._llm = llm
        self._search_mode = search_mode
        self.history = ChatHistory(max_turns=max_history_turns)

    def chat(self, query: str, limit: int = 5) -> ChatResponse:
        docs = self._search.search(query, limit=limit, mode=self._search_mode)
        messages = build_messages(query, docs, self.history.to_messages())
        answer = self._llm.generate(messages)
        self.history.add_turn(query=query, response=answer, sources=docs)
        return ChatResponse(answer=answer, sources=docs)

    def chat_stream(self, query: str, limit: int = 5) -> tuple[Iterator[str], list[dict]]:
        docs = self._search.search(query, limit=limit, mode=self._search_mode)
        messages = build_messages(query, docs, self.history.to_messages())

        def _stream_and_record():
            chunks = []
            for token in self._llm.stream(messages):
                chunks.append(token)
                yield token
            self.history.add_turn(query=query, response="".join(chunks), sources=docs)

        return _stream_and_record(), docs

    def clear_history(self) -> None:
        self.history.clear()
