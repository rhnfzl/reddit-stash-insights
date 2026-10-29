"""Embedders: local BGE-M3 (dense + sparse) or an OpenAI-compatible embeddings API (dense only)."""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from rsi.config import Settings

logger = logging.getLogger(__name__)

# Default model — configurable via constructor
DEFAULT_MODEL = "BAAI/bge-m3"


def _patch_transformers_compat() -> None:
    """Shim for FlagEmbedding <1.4 + transformers >=5.0 compatibility.

    ``transformers`` 5.x removed ``is_torch_fx_available`` which older
    FlagEmbedding versions import at module level in the reranker sub-package.
    Injecting a no-op stub prevents the ``ImportError`` without affecting
    any functionality we use (we only need the BGEM3FlagModel encoder).
    """
    try:
        from transformers.utils.import_utils import is_torch_fx_available  # noqa: F401
    except ImportError:
        import transformers.utils.import_utils as _tiu

        if not hasattr(_tiu, "is_torch_fx_available"):
            _tiu.is_torch_fx_available = lambda: False  # type: ignore[attr-defined]
            logger.debug("Injected is_torch_fx_available shim for transformers compat")


class Embedder:
    """Wrapper around BGE-M3 for generating dense and sparse embeddings."""

    provider = "local"

    def __init__(self, model_name: str = DEFAULT_MODEL, use_fp16: bool = True):
        self.model = model_name
        _patch_transformers_compat()
        from FlagEmbedding import BGEM3FlagModel

        logger.info("Loading embedding model: %s", model_name)
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self._model_name = model_name
        logger.info("Model loaded successfully")

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Embed a batch of texts, returning dense and sparse vectors.

        Returns:
            {"dense": list of 1024-dim vectors, "sparse": list of sparse weight dicts}
        """
        output = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": output["dense_vecs"].tolist(),
            "sparse": output["lexical_weights"],
        }

    def embed_query(self, query: str) -> Dict[str, Any]:
        """Embed a single query for search.

        Returns:
            {"dense": 1024-dim vector, "sparse": sparse weight dict}
        """
        result = self.embed([query])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }


class OpenAIEmbedder:
    """Dense embeddings from an OpenAI-compatible ``/v1/embeddings`` endpoint.

    Each input is truncated to ``max_input_tokens`` with the model tokenizer, and
    requests are packed under both ``batch_size`` items and ``max_batch_tokens``
    total tokens. Retries (429, 5xx, timeouts) are left to the SDK.
    """

    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 128,
        max_input_tokens: int = 8000,
        max_batch_tokens: int = 250_000,
        client: Any = None,
    ):
        import tiktoken

        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url, max_retries=5, timeout=60)
        self._client = client
        self.model = model
        self._batch_size = batch_size
        self._max_input_tokens = max_input_tokens
        self._max_batch_tokens = max_batch_tokens
        try:
            self._enc = tiktoken.encoding_for_model(model.split("/")[-1])
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def _prepare(self, text: str) -> tuple[str, int]:
        """Return the text truncated to the token cap and its token count."""
        tokens = self._enc.encode(text or " ", disallowed_special=())
        if len(tokens) <= self._max_input_tokens:
            return text or " ", len(tokens)  # the API rejects empty strings
        tokens = tokens[: self._max_input_tokens]
        return self._enc.decode(tokens), len(tokens)

    def _batches(self, prepared: List[tuple[str, int]]) -> List[List[str]]:
        batches: List[List[str]] = []
        batch: List[str] = []
        batch_tokens = 0
        for text, n in prepared:
            if batch and (len(batch) >= self._batch_size or batch_tokens + n > self._max_batch_tokens):
                batches.append(batch)
                batch, batch_tokens = [], 0
            batch.append(text)
            batch_tokens += n
        if batch:
            batches.append(batch)
        return batches

    def embed(self, texts: List[str]) -> Dict[str, Any]:
        """Embed texts in order. Returns {"dense": vectors, "sparse": empty dicts}."""
        dense: List[List[float]] = []
        for batch in self._batches([self._prepare(t) for t in texts]):
            resp = self._client.embeddings.create(model=self.model, input=batch)
            dense.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        if len(dense) != len(texts):
            raise RuntimeError(f"embeddings API returned {len(dense)} vectors for {len(texts)} inputs")
        return {"dense": dense, "sparse": [{} for _ in texts]}

    def embed_query(self, query: str) -> Dict[str, Any]:
        result = self.embed([query])
        return {"dense": result["dense"][0], "sparse": {}}


def make_embedder(settings: "Settings") -> Any:
    """Build the embedder selected by ``settings.embedding_provider``."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder(
            settings.embedding_model,
            base_url=settings.embedding_base_url,
            api_key=os.environ.get("RSI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            batch_size=settings.embedding_batch_size,
            max_input_tokens=settings.embedding_max_input_tokens,
            max_batch_tokens=settings.embedding_max_batch_tokens,
        )
    return Embedder(model_name=settings.embedding_model, use_fp16=settings.embedding_use_fp16)
