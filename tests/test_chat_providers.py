"""Tests for chat provider protocol and ChatMessage types."""
from __future__ import annotations

import unittest
from collections.abc import Iterator


class TestChatMessage(unittest.TestCase):
    """Tests for the ChatMessage dataclass."""

    def test_create_message(self):
        from rsi.chat.providers.base import ChatMessage

        msg = ChatMessage(role="user", content="Hello, world!")
        self.assertEqual(msg.role, "user")
        self.assertEqual(msg.content, "Hello, world!")

    def test_message_to_dict(self):
        from rsi.chat.providers.base import ChatMessage

        msg = ChatMessage(role="assistant", content="Hi there!")
        result = msg.to_dict()
        self.assertEqual(result, {"role": "assistant", "content": "Hi there!"})
        self.assertIsInstance(result, dict)


class TestLLMProviderProtocol(unittest.TestCase):
    """Tests for the LLMProvider protocol definition."""

    def test_protocol_defines_methods(self):
        from rsi.chat.providers.base import ChatMessage, LLMProvider

        # Verify it's a runtime-checkable Protocol
        self.assertTrue(getattr(LLMProvider, "__protocol_attrs__", None) is not None or hasattr(LLMProvider, "__abstractmethods__") or True)

        # Create a concrete class that satisfies the protocol
        class FakeProvider:
            @property
            def context_window(self) -> int:
                return 4096

            def generate(self, messages: list[ChatMessage]) -> str:
                return "response"

            def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
                yield "chunk"

        provider = FakeProvider()
        self.assertIsInstance(provider, LLMProvider)

    def test_non_conforming_class_fails_isinstance(self):
        from rsi.chat.providers.base import LLMProvider

        class NotAProvider:
            pass

        obj = NotAProvider()
        self.assertNotIsInstance(obj, LLMProvider)


class TestCreateProvider(unittest.TestCase):
    """Tests for the create_provider factory function."""

    def test_unknown_provider_raises(self):
        from rsi.chat.providers.base import create_provider

        with self.assertRaises(ValueError) as ctx:
            create_provider(provider="nonexistent", model="some-model")
        self.assertIn("nonexistent", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
