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
