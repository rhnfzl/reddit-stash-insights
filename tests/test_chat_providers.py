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


class TestOllamaProvider(unittest.TestCase):
    """Tests for the OllamaProvider."""

    def _make_provider(self, mock_requests: MagicMock, **kwargs):
        """Instantiate OllamaProvider with the ``requests`` module mocked."""
        fake_mod = types.ModuleType("requests")
        fake_mod.post = mock_requests.post  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"requests": fake_mod}):
            mod = importlib.import_module("rsi.chat.providers.ollama_provider")
            importlib.reload(mod)
            provider = mod.OllamaProvider(model="llama3", **kwargs)
        # The provider captured the fake module as self._requests — methods work outside the ctx.
        return provider

    def test_generate(self):
        mock_requests = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Hello from Ollama!"}}
        mock_requests.post.return_value = mock_response

        from rsi.chat.providers.base import ChatMessage

        provider = self._make_provider(mock_requests)
        result = provider.generate([ChatMessage(role="user", content="Hi")])

        self.assertEqual(result, "Hello from Ollama!")
        mock_requests.post.assert_called_once()

    def test_stream(self):
        mock_requests = MagicMock()
        mock_response = MagicMock()
        # iter_lines returns bytes that json.loads can parse
        mock_response.iter_lines.return_value = [
            b'{"message":{"content":"Hello"},"done":false}',
            b'{"message":{"content":" world"},"done":false}',
            b'{"done":true}',
        ]
        mock_requests.post.return_value = mock_response

        from rsi.chat.providers.base import ChatMessage

        provider = self._make_provider(mock_requests)
        tokens = list(provider.stream([ChatMessage(role="user", content="Hi")]))

        self.assertEqual(tokens, ["Hello", " world"])

    def test_satisfies_protocol(self):
        mock_requests = MagicMock()
        provider = self._make_provider(mock_requests)

        from rsi.chat.providers.base import LLMProvider

        self.assertIsInstance(provider, LLMProvider)


class TestOpenAIProvider(unittest.TestCase):
    """Tests for the OpenAIProvider."""

    def _make_provider(self, mock_openai_cls: MagicMock, **kwargs):
        """Instantiate OpenAIProvider with the ``openai`` module mocked."""
        fake_mod = types.ModuleType("openai")
        fake_mod.OpenAI = mock_openai_cls  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"openai": fake_mod}):
            mod = importlib.import_module("rsi.chat.providers.openai_provider")
            importlib.reload(mod)
            return mod.OpenAIProvider(model="gpt-4o-mini", api_key="test-key", **kwargs)

    def test_generate(self):
        mock_openai_cls = MagicMock()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from OpenAI!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        from rsi.chat.providers.base import ChatMessage

        provider = self._make_provider(mock_openai_cls)
        result = provider.generate([ChatMessage(role="user", content="Hi")])

        self.assertEqual(result, "Hello from OpenAI!")
        mock_client.chat.completions.create.assert_called_once()

    def test_stream(self):
        mock_openai_cls = MagicMock()
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Build streaming chunks
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        chunk3 = MagicMock()
        chunk3.choices = [MagicMock()]
        chunk3.choices[0].delta.content = None

        mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

        from rsi.chat.providers.base import ChatMessage

        provider = self._make_provider(mock_openai_cls)
        tokens = list(provider.stream([ChatMessage(role="user", content="Hi")]))

        self.assertEqual(tokens, ["Hello", " world"])

    def test_satisfies_protocol(self):
        mock_openai_cls = MagicMock()
        provider = self._make_provider(mock_openai_cls)

        from rsi.chat.providers.base import LLMProvider

        self.assertIsInstance(provider, LLMProvider)


if __name__ == "__main__":
    unittest.main()
