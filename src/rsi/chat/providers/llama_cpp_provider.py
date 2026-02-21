"""llama-cpp-python LLM provider -- loads GGUF models directly."""
from __future__ import annotations

from typing import Iterator

from rsi.chat.providers.base import ChatMessage


class LlamaCppProvider:
    """LLM provider using llama-cpp-python for local GGUF inference."""

    def __init__(
        self,
        model: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ):
        import os
        import sys

        from llama_cpp import Llama

        # Suppress C-level ggml_metal_init "skipping kernel" messages on stderr
        stderr_fd = sys.stderr.fileno()
        old_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        try:
            self._llm = Llama(
                model_path=model,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)
            os.close(devnull)
        self._n_ctx = n_ctx
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, messages: list[ChatMessage]) -> str:
        """Return a complete response for the given conversation."""
        response = self._llm.create_chat_completion(
            messages=[m.to_dict() for m in messages],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return response["choices"][0]["message"]["content"]

    def stream(self, messages: list[ChatMessage]) -> Iterator[str]:
        """Yield response tokens incrementally."""
        for chunk in self._llm.create_chat_completion(
            messages=[m.to_dict() for m in messages],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                yield delta["content"]

    @property
    def context_window(self) -> int:
        """Maximum token context length supported by the loaded model."""
        return self._n_ctx
