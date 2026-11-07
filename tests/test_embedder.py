"""Tests for BGE-M3 embedder.

NOTE: These tests download the BGE-M3 model (~2GB) on first run.
Skip with: RSI_SKIP_SLOW_TESTS=1
"""
import unittest
import os


SKIP_SLOW = os.environ.get("RSI_SKIP_SLOW_TESTS", "0") == "1"


class TestEmbedder(unittest.TestCase):
    @unittest.skipIf(SKIP_SLOW, "Skipping slow model download test")
    def test_embed_returns_dense_and_sparse(self):
        from rsi.indexer.embedder import Embedder

        embedder = Embedder()
        result = embedder.embed(["Hello world"])
        self.assertIn("dense", result)
        self.assertIn("sparse", result)
        self.assertEqual(len(result["dense"]), 1)
        self.assertEqual(len(result["dense"][0]), 1024)  # BGE-M3 dimensions

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model download test")
    def test_embed_batch(self):
        from rsi.indexer.embedder import Embedder

        embedder = Embedder()
        texts = ["First text", "Second text", "Third text"]
        result = embedder.embed(texts)
        self.assertEqual(len(result["dense"]), 3)
        self.assertEqual(len(result["sparse"]), 3)

    @unittest.skipIf(SKIP_SLOW, "Skipping slow model download test")
    def test_embed_query(self):
        from rsi.indexer.embedder import Embedder

        embedder = Embedder()
        result = embedder.embed_query("search query")
        self.assertIn("dense", result)
        self.assertEqual(len(result["dense"]), 1024)


if __name__ == "__main__":
    unittest.main()


class TestTokenizerAdapter(unittest.TestCase):
    def test_local_adapter_uses_active_model_tokenizer_without_special_tokens(self):
        from unittest.mock import Mock
        from rsi.indexer.embedder import Embedder
        embedder = object.__new__(Embedder)
        embedder._model = Mock()
        embedder._model.tokenizer.encode.return_value = [1, 2]
        embedder._model.tokenizer.decode.return_value = "source"
        self.assertEqual(embedder.encode("source"), [1, 2])
        self.assertEqual(embedder.decode([1, 2]), "source")
        embedder._model.tokenizer.encode.assert_called_once_with("source", add_special_tokens=False)

class TestLocalCapacity(unittest.TestCase):
    def make(self, tokenizer_limit, position_limit, special_tokens):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from rsi.indexer.embedder import Embedder
        embedder = object.__new__(Embedder)
        embedder._model = SimpleNamespace(
            tokenizer=SimpleNamespace(model_max_length=tokenizer_limit,
                                      num_special_tokens_to_add=Mock(return_value=special_tokens)),
            model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=position_limit)),
            passage_max_length=512,
            encode=Mock(return_value={"dense_vecs": SimpleNamespace(tolist=lambda: [[0.1]]),
                                      "lexical_weights": [{}]}),
        )
        return embedder

    def test_bge_uses_actual_model_capacity_with_special_token_allowance(self):
        embedder = self.make(8192, 8194, 2)
        self.assertEqual(embedder.max_input_tokens, 8190)
        embedder.embed(["passage"])
        self.assertEqual(embedder._model.encode.call_args.kwargs["max_length"], 8192)

    def test_smaller_model_limit_and_special_tokens_reduce_content_capacity(self):
        self.assertEqual(self.make(8192, 512, 2).max_input_tokens, 510)
