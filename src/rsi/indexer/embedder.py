"""BGE-M3 embedding model — produces dense + sparse vectors in one call."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

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

    def __init__(self, model_name: str = DEFAULT_MODEL, use_fp16: bool = True):
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
