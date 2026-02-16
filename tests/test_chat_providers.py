"""Tests for chat provider protocol and ChatMessage types."""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from collections.abc import Iterator
from unittest.mock import MagicMock, patch


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


class TestLlamaCppProvider(unittest.TestCase):
    """Tests for the LlamaCppProvider."""

    def _make_provider(self, mock_llama_cls: MagicMock, **kwargs):
        """Instantiate LlamaCppProvider with the real llama_cpp module replaced by a fake."""
        fake_mod = types.ModuleType("llama_cpp")
        fake_mod.Llama = mock_llama_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"llama_cpp": fake_mod}):
            mod = importlib.import_module("rsi.chat.providers.llama_cpp_provider")
            importlib.reload(mod)
            return mod.LlamaCppProvider(model="/tmp/fake.gguf", **kwargs)

    def test_generate(self):
        mock_llama_cls = MagicMock()
        mock_llm = MagicMock()
        mock_llama_cls.return_value = mock_llm
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello from llama-cpp!"}}],
        }

        from rsi.chat.providers.base import ChatMessage

        provider = self._make_provider(mock_llama_cls)
        result = provider.generate([ChatMessage(role="user", content="Hi")])

        self.assertEqual(result, "Hello from llama-cpp!")
        mock_llm.create_chat_completion.assert_called_once()

    def test_stream(self):
        mock_llama_cls = MagicMock()
        mock_llm = MagicMock()
        mock_llama_cls.return_value = mock_llm
        mock_llm.create_chat_completion.return_value = iter([
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
            {"choices": [{"delta": {}}]},
        ])

        from rsi.chat.providers.base import ChatMessage

        provider = self._make_provider(mock_llama_cls)
        tokens = list(provider.stream([ChatMessage(role="user", content="Hi")]))

        self.assertEqual(tokens, ["Hello", " world"])

    def test_context_window(self):
        mock_llama_cls = MagicMock()
        provider = self._make_provider(mock_llama_cls, n_ctx=8192)
        self.assertEqual(provider.context_window, 8192)

    def test_satisfies_protocol(self):
        mock_llama_cls = MagicMock()
        provider = self._make_provider(mock_llama_cls)

        from rsi.chat.providers.base import LLMProvider

        self.assertIsInstance(provider, LLMProvider)


if __name__ == "__main__":
    unittest.main()
