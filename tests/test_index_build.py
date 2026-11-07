"""Index build, metadata, compatibility checks and the health command (fake embedder, no model)."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

try:
    import lancedb  # noqa: F401
    HAS_LANCEDB = True
except ImportError:
    HAS_LANCEDB = False

POST = (
    "---\nid: {id}\nsubreddit: /r/{sub}\ntimestamp: 2024-01-01 00:00:00\n"
    "author: /u/test\ncomments: 0\n"
    "permalink: https://reddit.com/r/{sub}/comments/{id}/t/\n---\n\n"
    "# {title}\n\n**Upvotes:** 5 | **Permalink:** [Link](https://example.com)\n\n{body}\n"
)



def _meta(db: Path) -> dict:
    """read_meta for tests: fail loudly when the metadata file is missing."""
    from rsi.indexer.index_meta import read_meta

    meta = read_meta(db)
    assert meta is not None, f"no index metadata in {db}"
    assert isinstance(meta, dict), f"invalid index metadata in {db}"
    return meta


class FakeEmbedder:
    """Deterministic 8-dim vectors; counts calls; can refuse to be used at all."""

    max_input_tokens = 8000

    def __init__(self, provider="openai", model="text-embedding-3-small", forbid=False):
        self.provider, self.model, self.forbid = provider, model, forbid
        self.batches = []
        self.texts = []

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(tokens).decode("utf-8")

    def _vec(self, t):
        return [float((hash(t) >> i) & 0xFF) / 255.0 for i in range(8)]

    def embed(self, texts):
        if self.forbid:
            raise AssertionError("embedding call not allowed here")
        self.batches.append(len(texts))
        self.texts.extend(texts)
        return {"dense": [self._vec(t) for t in texts], "sparse": [{} for _ in texts]}

    def embed_query(self, q):
        return {"dense": self.embed([q])["dense"][0], "sparse": {}}


def make_reddit_dir(root: Path, n=5, broken=0) -> Path:
    d = root / "reddit" / "Python"
    d.mkdir(parents=True)
    for i in range(n):
        (d / f"POST_p{i}.md").write_text(
            POST.format(id=f"p{i}", sub="Python", title=f"Post {i} asyncio", body=f"zebrafish body number {i}")
        )
    for i in range(broken):
        (d / f"POST_bad{i}.md").write_text("---\nnot: [valid\n---\n")
    return root / "reddit"


class IndexBuildTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {"RSI_EMBEDDING_PROVIDER": "openai",
                                           "RSI_EMBEDDING_MODEL": "text-embedding-3-small"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def run_cli(self, args, embedder=None, env=None):
        from rsi.cli import app

        embedder = embedder or FakeEmbedder()
        with patch("rsi.indexer.search.make_embedder", return_value=embedder), \
                patch.dict(os.environ, env or {}):
            return CliRunner().invoke(app, args), embedder


@unittest.skipUnless(HAS_LANCEDB, "requires index extra (lancedb)")
class TestIndexBuild(IndexBuildTestCase):
    def test_fresh_openai_build_writes_metadata_last_and_honours_rsi_db_path(self):

        reddit = make_reddit_dir(self.root)
        db = self.root / "next"
        result, emb = self.run_cli(["index", str(reddit), "--max-error-pct", "1"],
                                   env={"RSI_DB_PATH": str(db), "RSI_INDEX_CHUNK": "2"})
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(emb.batches, [2, 2, 1])  # bounded chunks
        meta = _meta(db)
        self.assertEqual(meta["input_token_capacity"], 8000)
        self.assertEqual((meta["provider"], meta["model"], meta["dim"], meta["rows"], meta["schema"]),
                         ("openai", "text-embedding-3-small", 8, 5, 2))

    def test_fts_built_once_and_metadata_after_inserts(self):
        from rsi.indexer import index_meta
        from rsi.indexer.vector_store import VectorStore

        events = []
        orig_add, orig_fts, orig_write = VectorStore.add_records, VectorStore.create_fts_index, index_meta.write_meta

        def add(self_, records):
            events.append("add")
            return orig_add(self_, records)

        def fts(self_):
            events.append("fts")
            return orig_fts(self_)

        def write(*a, **kw):
            events.append("meta")
            return orig_write(*a, **kw)

        reddit = make_reddit_dir(self.root)
        with patch.object(VectorStore, "add_records", add), patch.object(VectorStore, "create_fts_index", fts), \
                patch("rsi.indexer.index_meta.write_meta", write):
            result, _ = self.run_cli(["index", str(reddit), "--db-path", str(self.root / "db")],
                                     env={"RSI_INDEX_CHUNK": "2"})
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Incomplete marker before the first row, completed metadata last.
        self.assertEqual(events, ["meta", "add", "add", "add", "fts", "meta"])

    def test_data_gate_zero_rows_exit_3_no_metadata(self):
        from rsi.indexer.index_meta import meta_path

        empty = self.root / "empty"
        empty.mkdir()
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(empty), "--db-path", str(db), "--max-error-pct", "1"])
        self.assertEqual(result.exit_code, 3, msg=result.output)
        self.assertFalse(meta_path(db).exists())

    def test_data_gate_parse_errors_over_threshold_exit_3(self):
        from rsi.indexer.index_meta import meta_path

        from rsi.core.scanner import scan_directory

        reddit = make_reddit_dir(self.root, n=5, broken=2)
        res = scan_directory(reddit)
        self.assertEqual((len(res.posts), len(res.errors)), (5, 2))  # items parse, so the zero-item gate is off
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db), "--max-error-pct", "1"])
        self.assertEqual(result.exit_code, 3, msg=result.output)
        self.assertFalse(meta_path(db).exists())
        # 2 of 7 files is 28.6%: a 50% limit lets the same archive through.
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db), "--max-error-pct", "50"])
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def _legacy_index(self, db):
        """A table with no metadata file: what every pre-metadata index looks like."""
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine

        engine = SearchEngine(db, embedder=FakeEmbedder("local", "BAAI/bge-m3"), settings=Settings())
        engine.index_texts(["legacy zebrafish row"], ["l1"], [{"subreddit": "x", "score": 0, "timestamp": "",
                                                               "file_path": "x.md", "content_type": "post",
                                                               "source": "own"}])

    def test_legacy_index_is_local_and_rejects_openai_reader(self):
        from rsi.config import Settings
        from rsi.indexer.index_meta import IndexConfigMismatch
        from rsi.indexer.search import SearchEngine

        db = self.root / "db"
        self._legacy_index(db)
        SearchEngine(db, settings=Settings())  # local reader: fine
        with self.assertRaises(IndexConfigMismatch) as cm:
            SearchEngine(db, settings=Settings(embedding_provider="openai", embedding_model="text-embedding-3-small"))
        self.assertIn("--rebuild", str(cm.exception))

    def test_absent_index_is_compatible_with_anything(self):
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine

        SearchEngine(self.root / "fresh", settings=Settings(embedding_provider="openai", embedding_model="m"))

    def test_index_over_legacy_needs_rebuild(self):
        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        self._legacy_index(db)
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 4, msg=result.output)
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db), "--rebuild"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(_meta(db)["rows"], 5)  # legacy row dropped, not appended to

    def test_search_cli_mismatch_exit_4(self):
        db = self.root / "db"
        self._legacy_index(db)
        result, _ = self.run_cli(["search", "zebrafish", "--db-path", str(db), "--mode", "keyword"])
        self.assertEqual(result.exit_code, 4, msg=result.output)
        self.assertIn("rebuild", result.output)

    def test_index_retains_late_evidence_in_bounded_passages(self):
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine

        reddit = make_reddit_dir(self.root, n=1)
        path = reddit / "Python" / "POST_p0.md"
        path.write_text(POST.format(id="p0", sub="Python", title="Late thread",
                                    body="early " * 6000 + "late zebrafish evidence"))
        db = self.root / "db"
        result, emb = self.run_cli(["index", str(reddit), "--db-path", str(db)],
                                   env={"RSI_MAX_TEXT_CHARS": "12"})
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertTrue(emb.texts and all(len(t.encode()) <= 512 for t in emb.texts))
        self.assertTrue(any("late zebrafish evidence" in t for t in emb.texts))
        engine = SearchEngine(db, embedder=FakeEmbedder(), settings=Settings())
        rows = engine.search("zebrafish", mode=__import__("rsi.indexer.search", fromlist=["SearchMode"]).SearchMode.KEYWORD)
        self.assertTrue(rows and "late zebrafish evidence" in rows[0]["text"])

    def test_max_text_chars_must_be_positive(self):
        from rsi.config import Settings

        with patch.dict(os.environ, {"RSI_MAX_TEXT_CHARS": "0"}):
            with self.assertRaises(ValueError):
                Settings.load()

    def test_health_keyword_ok_without_embedding_call(self):
        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        self.run_cli(["index", str(reddit), "--db-path", str(db)])
        result, _ = self.run_cli(["health"], embedder=FakeEmbedder(forbid=True), env={"RSI_DB_PATH": str(db)})
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("healthy", result.output)

    def test_health_hybrid_uses_embedder(self):
        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        self.run_cli(["index", str(reddit), "--db-path", str(db)])
        result, emb = self.run_cli(["health", "--db-path", str(db), "--mode", "hybrid"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual(emb.batches, [1])

    def test_health_missing_index_exit_1(self):
        result, _ = self.run_cli(["health", "--db-path", str(self.root / "nope")])
        self.assertEqual(result.exit_code, 1)

    def test_health_mismatch_exit_4(self):
        db = self.root / "db"
        self._legacy_index(db)
        result, _ = self.run_cli(["health", "--db-path", str(db)])
        self.assertEqual(result.exit_code, 4, msg=result.output)

    def _index_rows(self, db, texts):
        from rsi.config import Settings
        from rsi.indexer.index_meta import write_meta
        from rsi.indexer.search import SearchEngine

        engine = SearchEngine(db, embedder=FakeEmbedder(), settings=Settings())
        meta = {"subreddit": "x", "score": 0, "timestamp": "", "file_path": "x.md", "content_type": "post",
                "source": "own"}
        engine.index_texts(texts, [f"r{i}" for i in range(len(texts))], [dict(meta) for _ in texts])
        write_meta(db, provider="openai", model="text-embedding-3-small", dim=8, rows=len(texts), parse_errors=0)

    def test_health_probe_skips_stop_words(self):
        db = self.root / "db"
        self._index_rows(db, ["The Python guide"])
        result, _ = self.run_cli(["health", "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_health_probe_many_long_stop_words(self):
        db = self.root / "db"
        self._index_rows(db, ["against between through about after before during ok"])
        result, _ = self.run_cli(["health", "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_health_probe_short_text_and_long_identifier(self):
        db = self.root / "db"
        self._index_rows(db, ["ok", "an_identifier_" + "x" * 60 + " ok"])
        result, _ = self.run_cli(["health", "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_crashed_build_is_rejected_by_readers_and_resumable(self):

        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        calls = {"n": 0}

        class FailSecondChunk(FakeEmbedder):
            def embed(self, texts):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise RuntimeError("embeddings API down")
                return super().embed(texts)

        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)], embedder=FailSecondChunk(),
                                 env={"RSI_INDEX_CHUNK": "2"})
        self.assertNotEqual(result.exit_code, 0)
        self.assertIs(_meta(db)["complete"], False)
        for args in (["health", "--db-path", str(db)], ["search", "zebrafish", "--db-path", str(db), "-m", "keyword"]):
            result, _ = self.run_cli(args)
            self.assertEqual(result.exit_code, 1, msg=result.output)
            self.assertIn("unfinished", result.output)
        # Same identity as the legacy rule would give, so the old "table without metadata" path cannot hide it.
        result, _ = self.run_cli(["health", "--db-path", str(db)], env={"RSI_EMBEDDING_PROVIDER": "local",
                                                                       "RSI_EMBEDDING_MODEL": "BAAI/bge-m3"})
        self.assertNotEqual(result.exit_code, 0, msg=result.output)
        # Retrying without --rebuild finishes the build with no duplicate rows.
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)], env={"RSI_INDEX_CHUNK": "2"})
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual((_meta(db)["complete"], _meta(db)["rows"]), (True, 5))
        result, _ = self.run_cli(["health", "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, msg=result.output)

    def test_http_startup_rejects_crashed_build(self):
        from rsi import mcp_server
        from rsi.indexer.index_meta import IndexIncomplete, write_meta

        db = self.root / "db"
        self._index_rows(db, ["zebrafish row"])
        write_meta(db, provider="openai", model="text-embedding-3-small", dim=0, rows=0, parse_errors=0,
                   complete=False)
        with patch.dict(os.environ, {"RSI_MCP_TOKEN": "t", "RSI_DB_PATH": str(db)}), \
                patch("uvicorn.run") as run, self.assertRaises(IndexIncomplete):
            mcp_server.run_http("127.0.0.1", 0)
        run.assert_not_called()

    def test_reindex_is_idempotent(self):
        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        for _ in range(2):
            result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
            self.assertEqual(result.exit_code, 0, msg=result.output)
        from rsi.indexer.vector_store import VectorStore
        self.assertEqual(VectorStore(db).count(), 5)

    def test_non_positive_chunk_size_rejected_before_index_changes(self):
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine

        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        self.run_cli(["index", str(reddit), "--db-path", str(db)])
        before = (db / "rsi_index.json").read_text()
        result, emb = self.run_cli(["index", str(reddit), "--db-path", str(db)], env={"RSI_INDEX_CHUNK": "-1"})
        self.assertNotEqual(result.exit_code, 0, msg=result.output)
        self.assertEqual((db / "rsi_index.json").read_text(), before)
        self.assertEqual(emb.batches, [])
        engine = SearchEngine(db, embedder=FakeEmbedder(), settings=Settings())
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                engine.index_texts(["a"], ["a"], [{}], chunk_size=bad)

    def test_concurrent_first_searches_build_one_embedder(self):
        import threading
        import time

        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine, SearchMode

        db = self.root / "db"
        self._index_rows(db, ["zebrafish row"])
        engine = SearchEngine(db, settings=Settings(embedding_provider="openai",
                                                    embedding_model="text-embedding-3-small"))
        built = []

        def slow_make(_settings):
            time.sleep(0.05)  # widen the race window
            built.append(FakeEmbedder())
            return built[-1]

        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            engine.search("zebrafish", limit=1, mode=SearchMode.SEMANTIC)

        with patch("rsi.indexer.search.make_embedder", side_effect=slow_make):
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(built), 1)

    def test_concurrent_first_mcp_calls_open_one_engine(self):
        import threading
        import time

        from rsi import mcp_server

        opened = []

        def slow_engine(*a, **kw):
            time.sleep(0.05)
            opened.append(object())
            return opened[-1]

        barrier = threading.Barrier(4)

        def worker():
            barrier.wait()
            mcp_server._get_search_engine()

        prev, mcp_server._search_engine = mcp_server._search_engine, None
        try:
            with patch("rsi.indexer.search.SearchEngine", side_effect=slow_engine):
                threads = [threading.Thread(target=worker) for _ in range(4)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
        finally:
            mcp_server._search_engine = prev
        self.assertEqual(len(opened), 1)

    def test_shortened_source_removes_old_passages_only_after_success(self):
        from rsi.indexer.vector_store import VectorStore

        reddit = make_reddit_dir(self.root, n=1)
        path = reddit / "Python" / "POST_p0.md"
        path.write_text(POST.format(id="p0", sub="Python", title="Title", body="long evidence " * 400))
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        before = VectorStore(db).count()
        self.assertGreater(before, 5)
        path.write_text(POST.format(id="p0", sub="Python", title="Title", body="short zebrafish"))
        class Failure(FakeEmbedder):
            def embed(self, texts):
                raise RuntimeError("embedding failed")
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)], embedder=Failure())
        self.assertNotEqual(result.exit_code, 0)
        self.assertEqual(VectorStore(db).count(), before)
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(VectorStore(db).count(), 1)
        self.assertTrue(_meta(db)["complete"])

    def test_unknown_schema_and_changed_chunk_config_reject_reader_and_writer(self):
        from rsi.config import Settings
        from rsi.indexer.index_meta import IndexConfigMismatch
        from rsi.indexer.search import SearchEngine

        reddit = make_reddit_dir(self.root, n=1)
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        with self.assertRaises(IndexConfigMismatch):
            SearchEngine(db, embedder=FakeEmbedder(), settings=Settings(chunk_tokens=256))
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)],
                                 env={"RSI_CHUNK_TOKENS": "256"})
        self.assertEqual(result.exit_code, 4, result.output)
        meta = _meta(db)
        meta["schema"] = 999
        (db / "rsi_index.json").write_text(json.dumps(meta))
        with self.assertRaises(IndexConfigMismatch):
            SearchEngine(db, embedder=FakeEmbedder(), settings=Settings())

    def test_unreleased_previous_chunk_algorithm_requires_fresh_index(self):
        from rsi.config import Settings
        from rsi.indexer.index_meta import IndexConfigMismatch
        from rsi.indexer.search import SearchEngine
        reddit = make_reddit_dir(self.root, n=1)
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        meta = _meta(db)
        self.assertEqual(meta["chunks"]["version"], 3)
        meta["chunks"]["version"] = 2
        (db / "rsi_index.json").write_text(json.dumps(meta))
        before = (db / "rsi_index.json").read_bytes()
        with self.assertRaises(IndexConfigMismatch):
            SearchEngine(db, embedder=FakeEmbedder(), settings=Settings())
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 4, result.output)
        self.assertEqual((db / "rsi_index.json").read_bytes(), before)

    def test_every_serialized_passage_has_fixed_nonempty_evidence_structs(self):
        from rsi.indexer.vector_store import VectorStore
        reddit = make_reddit_dir(self.root, n=1)
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        store = VectorStore(db)
        initial = store._table.search().limit(1).to_list()[0]
        self.assertTrue(initial["evidence_refs"])
        self.assertEqual(set(initial["evidence_refs"][0]),
                         {"evidence_id", "permalink", "parent_id", "source_start", "source_end"})
        path = reddit / "Python" / "POST_p0.md"
        with path.open("a") as target:
            target.write("\n**Comment by /u/reply**\n*Upvotes: 1 | "
                         "[Permalink](https://reddit.com/r/Python/comments/p0/title/c1/)*\n\nshort reply\n")
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        rows = VectorStore(db)._table.search().limit(10).to_list()
        self.assertTrue(any(len(row["evidence_refs"]) > 1 for row in rows))

    def test_schema_one_index_remains_readable_but_writer_requires_rebuild(self):
        from rsi.config import Settings
        from rsi.indexer.search import SearchEngine
        db = self.root / "db"
        self._index_rows(db, ["legacy zebrafish"])
        meta = _meta(db)
        meta["schema"] = 1
        meta.pop("chunks")
        (db / "rsi_index.json").write_text(json.dumps(meta))
        engine = SearchEngine(db, embedder=FakeEmbedder(), settings=Settings())
        self.assertEqual(engine.store.count(), 1)
        reddit = make_reddit_dir(self.root, n=1)
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 4, result.output)
        self.assertEqual(engine.store.count(), 1)

    def test_metadata_file_location(self):
        from rsi.indexer.index_meta import meta_path

        reddit = make_reddit_dir(self.root)
        db = self.root / "db"
        self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(meta_path(db), db / "rsi_index.json")
        self.assertEqual(json.loads((db / "rsi_index.json").read_text())["provider"], "openai")


if __name__ == "__main__":
    unittest.main()

@unittest.skipUnless(HAS_LANCEDB, "requires index extra (lancedb)")
class TestPassageCapacity(IndexBuildTestCase):
    def test_actual_openai_capacity_rejected_before_any_index_write(self):
        from rsi.indexer.embedder import OpenAIEmbedder
        from tests.test_openai_embedder import HAS_DEPS, fake_client
        if not HAS_DEPS:
            self.skipTest("requires tiktoken")
        reddit = make_reddit_dir(self.root, n=1)
        for capacity, requested in ((8000, 9000), (100, 512)):
            with self.subTest(capacity=capacity, requested=requested):
                db = self.root / f"capacity-{capacity}"
                client = fake_client()
                embedder = OpenAIEmbedder("text-embedding-3-small", client=client, max_input_tokens=capacity)
                result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)], embedder=embedder,
                                         env={"RSI_CHUNK_TOKENS": str(requested)})
                self.assertNotEqual(result.exit_code, 0, result.output)
                self.assertIn("capacity", result.output)
                self.assertFalse((db / "rsi_index.json").exists())
                from rsi.indexer.vector_store import VectorStore
                self.assertEqual(VectorStore(db).count(), 0)
                self.assertEqual(client.embeddings.calls, [])

    def test_capacity_rejected_before_rebuild_discards_old_index(self):
        reddit = make_reddit_dir(self.root, n=1)
        db = self.root / "db"
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db)])
        self.assertEqual(result.exit_code, 0, result.output)
        before = (db / "rsi_index.json").read_bytes()
        embedder = FakeEmbedder()
        embedder.max_input_tokens = 100
        result, _ = self.run_cli(["index", str(reddit), "--db-path", str(db), "--rebuild"], embedder=embedder)
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertEqual((db / "rsi_index.json").read_bytes(), before)
        from rsi.indexer.vector_store import VectorStore
        self.assertEqual(VectorStore(db).count(), 1)
