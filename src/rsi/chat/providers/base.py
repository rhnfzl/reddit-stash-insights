"""Base protocol and types for LLM providers."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A single message in a chat conversation."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialize to the ``{"role": ..., "content": ...}`` format expected by LLM APIs."""
        return {"role": self.role, "content": self.content}


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that every LLM backend must satisfy."""

    @property
    def context_window(self) -> int:
        """Maximum token context length supported by the loaded model."""
        ...

    def generate(self, messages: list[ChatMessage]) -> str:
        """Return a complete response for the given conversation."""
        ...

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield response tokens incrementally."""
        ...


def create_provider(provider: str, model: str, **kwargs: object) -> LLMProvider:
    """Instantiate an :class:`LLMProvider` by name.

    Parameters
    ----------
    provider:
        Backend identifier — one of ``"llama-cpp"``, ``"ollama"``, ``"openai"``.
    model:
        Model path (llama-cpp GGUF), model tag (ollama), or model name (openai).
    **kwargs:
        Extra keyword arguments forwarded to the provider constructor.

    Raises
    ------
    ValueError
        If *provider* is not a recognised backend name.
    """
    name = provider.lower().strip()

    if name == "llama-cpp":
        from rsi.chat.providers.llama_cpp_provider import LlamaCppProvider  # type: ignore[import-not-found]

        return LlamaCppProvider(model=model, **kwargs)

    if name == "ollama":
        from rsi.chat.providers.ollama_provider import OllamaProvider  # type: ignore[import-not-found]

        return OllamaProvider(model=model, **kwargs)

    if name == "openai":
        from rsi.chat.providers.openai_provider import OpenAIProvider  # type: ignore[import-not-found]

        return OpenAIProvider(model=model, **kwargs)

    msg = f"Unknown LLM provider: {provider!r}. Supported: 'llama-cpp', 'ollama', 'openai'."
    raise ValueError(msg)
