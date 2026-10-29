"""Tests for the OpenAI-compatible embedder (fake client, no network)."""
import os
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import tiktoken  # noqa: F401
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


class FakeEmbeddings:
    def __init__(self, dim=4, shuffle=False):
        self.calls = []
        self.dim = dim
        self.shuffle = shuffle

    def create(self, model, input):
        self.calls.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(len(t))] * self.dim) for i, t in enumerate(input)]
        if self.shuffle:
            data.reverse()
        return SimpleNamespace(data=data)


def fake_client(**kw):
    return SimpleNamespace(embeddings=FakeEmbeddings(**kw))


def token_dense(n_chars, seed=0):
    """CJK plus random identifiers: well over one token per character for cl100k."""
    rng = random.Random(seed)
    alphabet = "汉字測試데이터ü€" + "".join(chr(c) for c in range(0x4E00, 0x4E00 + 200))
    return "".join(rng.choice(alphabet) for _ in range(n_chars))


@unittest.skipUnless(HAS_DEPS, "requires embed-openai extra (tiktoken)")
class TestOpenAIEmbedder(unittest.TestCase):
    def make(self, **kw):
        from rsi.indexer.embedder import OpenAIEmbedder

        client = kw.pop("client", None) or fake_client()
        return OpenAIEmbedder("text-embedding-3-small", client=client, **kw), client.embeddings

    def test_token_dense_input_truncated_by_tokens_not_chars(self):
        emb, fake = self.make(max_input_tokens=100)
        text = token_dense(500)
        self.assertGreater(len(emb._enc.encode(text)), 100)
        emb.embed([text])
        sent = fake.calls[0][0]
        self.assertLessEqual(len(emb._enc.encode(sent)), 101)  # decode/re-encode can shift by one at the cut

    def test_query_is_truncated_too(self):
        emb, fake = self.make(max_input_tokens=50)
        out = emb.embed_query(token_dense(400, seed=1))
        self.assertLessEqual(len(emb._enc.encode(fake.calls[0][0])), 51)
        self.assertEqual(out["sparse"], {})
        self.assertEqual(len(out["dense"]), 4)

    def test_batches_bounded_by_item_count(self):
        emb, fake = self.make(batch_size=128)
        out = emb.embed([f"short text {i}" for i in range(300)])
        self.assertEqual([len(c) for c in fake.calls], [128, 128, 44])
        self.assertEqual(len(out["dense"]), 300)

    def test_batches_bounded_by_aggregate_tokens(self):
        # 10 inputs of ~3000 tokens with a 10k-token request budget: never more than 3 per request.
        emb, fake = self.make(batch_size=128, max_input_tokens=3000, max_batch_tokens=10_000)
        texts = ["word " * 4000 for _ in range(10)]
        emb.embed(texts)
        for call in fake.calls:
            self.assertLessEqual(sum(len(emb._enc.encode(t)) for t in call), 10_000)
        self.assertEqual(sum(len(c) for c in fake.calls), 10)
        self.assertGreater(len(fake.calls), 3)

    def test_order_kept_when_api_returns_out_of_order(self):
        emb, _ = self.make(client=fake_client(shuffle=True))
        out = emb.embed(["a", "bbb", "cc"])
        self.assertEqual([v[0] for v in out["dense"]], [1.0, 3.0, 2.0])

    def test_empty_text_is_not_sent_empty(self):
        emb, fake = self.make()
        emb.embed([""])
        self.assertEqual(fake.calls[0], [" "])

    def test_make_embedder_openai(self):
        from rsi.config import Settings
        from rsi.indexer.embedder import OpenAIEmbedder, make_embedder

        settings = Settings(embedding_provider="openai", embedding_model="text-embedding-3-small",
                            embedding_base_url="http://llm.example:4000/v1")
        with patch.dict(os.environ, {"RSI_EMBEDDING_API_KEY": "k"}), patch("openai.OpenAI") as client_cls:
            emb = make_embedder(settings)
        self.assertIsInstance(emb, OpenAIEmbedder)
        kwargs = client_cls.call_args.kwargs
        self.assertEqual(kwargs["base_url"], "http://llm.example:4000/v1")
        self.assertEqual(kwargs["api_key"], "k")
        self.assertEqual(kwargs["max_retries"], 5)


class TestEmbeddingSettings(unittest.TestCase):
    def load(self, env):
        from pathlib import Path
        import tempfile

        from rsi.config import Settings

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, env):
            return Settings.load(config_path=Path(tmp) / "none.toml")

    def test_default_is_local(self):
        with patch.dict(os.environ, {}, clear=False):
            for k in ("RSI_EMBEDDING_PROVIDER", "RSI_EMBEDDING_MODEL"):
                os.environ.pop(k, None)
            s = self.load({})
        self.assertEqual((s.embedding_provider, s.embedding_model), ("local", "BAAI/bge-m3"))

    def test_openai_gets_openai_default_model(self):
        s = self.load({"RSI_EMBEDDING_PROVIDER": "openai"})
        self.assertEqual(s.embedding_model, "text-embedding-3-small")

    def test_invalid_provider_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"RSI_EMBEDDING_PROVIDER": "bogus"})


if __name__ == "__main__":
    unittest.main()
