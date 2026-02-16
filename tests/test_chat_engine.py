"""Tests for RAG chat engine: prompt templates, history, and DirectEngine."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from rsi.chat.engine import ChatResponse, DirectEngine
from rsi.chat.history import ChatHistory
from rsi.chat.prompt import SYSTEM_PROMPT, build_context_block, build_messages
from rsi.chat.providers.base import ChatMessage


class TestPrompt(unittest.TestCase):
    """Tests for prompt template functions."""

    def test_build_context_block(self) -> None:
        docs = [
            {"subreddit": "python", "score": 42, "content_type": "post", "text": "Hello world"},
            {"subreddit": "rust", "score": 99, "content_type": "comment", "text": "Goodbye world"},
        ]
        result = build_context_block(docs)
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
        self.assertIn("r/python", result)
        self.assertIn("r/rust", result)
        self.assertIn("Hello world", result)
        self.assertIn("Goodbye world", result)

    def test_build_context_block_truncates_long_text(self) -> None:
        long_text = "x" * 1000
        docs = [{"subreddit": "test", "score": 1, "content_type": "post", "text": long_text}]
        result = build_context_block(docs, max_text_len=500)
        # The text in the block should be at most 500 chars
        lines = result.split("\n", 1)
        text_portion = lines[1]  # Everything after the header line
        self.assertEqual(len(text_portion), 500)

    def test_build_messages_single_turn(self) -> None:
        docs = [{"subreddit": "python", "score": 10, "content_type": "post", "text": "doc text"}]
        messages = build_messages("What is Python?", docs, history=[])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, SYSTEM_PROMPT)
        self.assertEqual(messages[-1].role, "user")
        self.assertIn("What is Python?", messages[-1].content)
        self.assertIn("doc text", messages[-1].content)

    def test_build_messages_with_history(self) -> None:
        docs = [{"subreddit": "test", "score": 1, "content_type": "post", "text": "ctx"}]
        history = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello!"),
        ]
        messages = build_messages("Follow up?", docs, history=history)
        self.assertEqual(len(messages), 4)  # system + 2 history + user
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[1].role, "user")
        self.assertEqual(messages[1].content, "Hi")
        self.assertEqual(messages[2].role, "assistant")
        self.assertEqual(messages[2].content, "Hello!")
        self.assertEqual(messages[3].role, "user")
        self.assertIn("Follow up?", messages[3].content)


class TestChatHistory(unittest.TestCase):
    """Tests for ChatHistory conversation manager."""

    def test_empty_history(self) -> None:
        h = ChatHistory()
        self.assertEqual(len(h), 0)
        self.assertEqual(h.to_messages(), [])

    def test_add_turn(self) -> None:
        h = ChatHistory()
        h.add_turn(query="q1", response="r1", sources=[{"id": "1"}])
        self.assertEqual(len(h), 1)
        msgs = h.to_messages()
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], ChatMessage(role="user", content="q1"))
        self.assertEqual(msgs[1], ChatMessage(role="assistant", content="r1"))

    def test_max_turns_eviction(self) -> None:
        h = ChatHistory(max_turns=2)
        h.add_turn(query="q1", response="r1", sources=[])
        h.add_turn(query="q2", response="r2", sources=[])
        h.add_turn(query="q3", response="r3", sources=[])
        self.assertEqual(len(h), 2)
        msgs = h.to_messages()
        # Oldest (q1) should be evicted
        self.assertEqual(msgs[0].content, "q2")
        self.assertEqual(msgs[2].content, "q3")

    def test_clear(self) -> None:
        h = ChatHistory()
        h.add_turn(query="q", response="r", sources=[])
        h.clear()
        self.assertEqual(len(h), 0)

    def test_last_sources(self) -> None:
        h = ChatHistory()
        h.add_turn(query="q1", response="r1", sources=[{"id": "a"}])
        h.add_turn(query="q2", response="r2", sources=[{"id": "b"}])
        self.assertEqual(h.last_sources(), [{"id": "b"}])

    def test_last_sources_empty(self) -> None:
        h = ChatHistory()
        self.assertEqual(h.last_sources(), [])

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.json"
            h = ChatHistory(max_turns=5)
            h.add_turn(query="q1", response="r1", sources=[{"id": "1"}])
            h.add_turn(query="q2", response="r2", sources=[{"id": "2"}])
            h.save(path)

            loaded = ChatHistory.load(path, max_turns=5)
            self.assertEqual(len(loaded), 2)
            msgs = loaded.to_messages()
            self.assertEqual(msgs[0].content, "q1")
            self.assertEqual(msgs[1].content, "r1")
            self.assertEqual(msgs[2].content, "q2")
            self.assertEqual(msgs[3].content, "r2")


class TestChatResponse(unittest.TestCase):
    """Tests for ChatResponse dataclass."""

    def test_create_response(self) -> None:
        resp = ChatResponse(answer="hello", sources=[{"id": "1"}])
        self.assertEqual(resp.answer, "hello")
        self.assertEqual(resp.sources, [{"id": "1"}])


class TestDirectEngine(unittest.TestCase):
    """Tests for the DirectEngine RAG pipeline."""

    def _make_engine(
        self,
        search_results: list[dict] | None = None,
        llm_response: str = "Mock answer",
    ) -> tuple[DirectEngine, MagicMock, MagicMock]:
        """Create a DirectEngine with mocked SearchEngine and LLM."""
        mock_search = MagicMock()
        mock_search.search.return_value = search_results or [
            {"subreddit": "python", "score": 10, "content_type": "post", "text": "Python is great"},
        ]

        mock_llm = MagicMock()
        mock_llm.generate.return_value = llm_response

        engine = DirectEngine(
            search_engine=mock_search,
            llm=mock_llm,
        )
        return engine, mock_search, mock_llm

    def test_chat_returns_response(self) -> None:
        engine, _, _ = self._make_engine(llm_response="Python is a language [1].")
        resp = engine.chat("What is Python?")
        self.assertIsInstance(resp, ChatResponse)
        self.assertEqual(resp.answer, "Python is a language [1].")
        self.assertEqual(len(resp.sources), 1)
        self.assertEqual(resp.sources[0]["subreddit"], "python")

    def test_chat_calls_search(self) -> None:
        engine, mock_search, _ = self._make_engine()
        engine.chat("query", limit=3)
        mock_search.search.assert_called_once()
        call_kwargs = mock_search.search.call_args
        self.assertEqual(call_kwargs.kwargs.get("limit") or call_kwargs[1].get("limit"), 3)

    def test_chat_passes_messages_to_llm(self) -> None:
        engine, _, mock_llm = self._make_engine()
        engine.chat("What is Python?")
        mock_llm.generate.assert_called_once()
        messages = mock_llm.generate.call_args[0][0]
        # First message is system prompt
        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, SYSTEM_PROMPT)
        # Last message is user query with context
        self.assertEqual(messages[-1].role, "user")
        self.assertIn("What is Python?", messages[-1].content)
        self.assertIn("Python is great", messages[-1].content)

    def test_chat_builds_history(self) -> None:
        engine, _, _ = self._make_engine()
        engine.chat("First question")
        engine.chat("Second question")
        msgs = engine.history.to_messages()
        self.assertEqual(len(msgs), 4)  # 2 turns x 2 messages each

    def test_clear_history(self) -> None:
        engine, _, _ = self._make_engine()
        engine.chat("question")
        self.assertEqual(len(engine.history), 1)
        engine.clear_history()
        self.assertEqual(len(engine.history), 0)


if __name__ == "__main__":
    unittest.main()
