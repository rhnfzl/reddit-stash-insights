"""OpenAI-compatible LLM provider -- works with OpenAI, vLLM, LM Studio."""
from __future__ import annotations

import os
from typing import Iterator

from rsi.chat.providers.base import ChatMessage


class OpenAIProvider:
    """LLM provider using the OpenAI chat completions API."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
        context_window: int = 8192,
    ):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_window = context_window

    def generate(self, messages: list[ChatMessage]) -> str:
        """Return a complete response for the given conversation."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield response tokens incrementally."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_dict() for m in messages],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    @property
    def context_window(self) -> int:
        """Maximum token context length supported by the loaded model."""
        return self._context_window
