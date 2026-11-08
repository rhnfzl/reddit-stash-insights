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
        if not 0 < max_tokens < n_ctx:
            raise ValueError("max_tokens must be positive and smaller than n_ctx")
        import os
        import sys

        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import (
            Jinja2ChatFormatter,
            chat_formatter_to_chat_completion_handler,
            format_llama2,
        )

        # Suppress C-level ggml_metal_init "skipping kernel" messages on stderr
        stderr_fd = sys.stderr.fileno()
        old_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, stderr_fd)
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
        self._n_ctx = self._llm.n_ctx()
        self._temperature = temperature
        self._max_tokens = max_tokens
        template = self._llm.metadata.get("tokenizer.chat_template")
        eos_id, bos_id = self._llm.token_eos(), self._llm.token_bos()
        if isinstance(template, str) and eos_id < 0:
            raise ValueError("GGUF chat template requires an EOS token; this model is unsupported")
        self._formatter = (
            Jinja2ChatFormatter(
                template=template,
                eos_token=self._llm.detokenize([eos_id], special=True).decode("utf-8") if eos_id >= 0 else "",
                bos_token=self._llm.detokenize([bos_id], special=True).decode("utf-8") if bos_id >= 0 else "",
                stop_token_ids=[eos_id] if eos_id >= 0 else None,
            ) if isinstance(template, str) else format_llama2
        )
        # Counting and generation must render the same template, including stop tokens.
        self._llm.chat_handler = chat_formatter_to_chat_completion_handler(self._formatter)

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

    @property
    def max_output_tokens(self) -> int:
        return self._max_tokens

    def count_tokens(self, messages: list[ChatMessage]) -> int:
        rendered = self._formatter(messages=[m.to_dict() for m in messages])
        return len(self._llm.tokenize(rendered.prompt.encode("utf-8"),
                                      add_bos=not rendered.added_special, special=True))
