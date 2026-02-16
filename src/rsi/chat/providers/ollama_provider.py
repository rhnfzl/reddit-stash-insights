"""Ollama LLM provider -- HTTP client to local Ollama server."""
from __future__ import annotations

import json
from typing import Iterator

from rsi.chat.providers.base import ChatMessage


class OllamaProvider:
    """LLM provider using Ollama HTTP API (localhost:11434)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        max_tokens: int = 512,
        context_window: int = 4096,
    ):
        import requests

        self._requests = requests
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_window = context_window

    def _build_payload(self, messages: list[ChatMessage], *, stream: bool) -> dict:
        return {
            "model": self._model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }

    def generate(self, messages: list[ChatMessage]) -> str:
        """Return a complete response for the given conversation."""
        response = self._requests.post(
            f"{self._base_url}/api/chat",
            json=self._build_payload(messages, stream=False),
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield response tokens incrementally."""
        response = self._requests.post(
            f"{self._base_url}/api/chat",
            json=self._build_payload(messages, stream=True),
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if data.get("done"):
                break
            content = data.get("message", {}).get("content", "")
            if content:
                yield content

    @property
    def context_window(self) -> int:
        """Maximum token context length supported by the loaded model."""
        return self._context_window
