"""CLI evaluation runs locally and errors cannot become successful scores."""
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from rsi.cli import app


class TestEvalCommand(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("lancedb"), "index extra not installed")
    def test_raw_engine_candidates_fetch_only_the_pool_needed_for_ten_threads(self):
        from rsi.evaluation import retrieve_threads
        from rsi.indexer.search import SearchEngine, SearchMode

        engine = SearchEngine.__new__(SearchEngine)
        store = MagicMock()
        engine._store = store
        store.fts_search.side_effect = lambda query, limit, subreddit: [
            {"id": f"chunk{i}", "content_type": "post", "parent_doc_id": f"post:p{i // 12}"}
            for i in range(min(limit, 120))
        ]

        def search(query, limit, mode, subreddit):
            return engine.search_candidates(query, limit=limit, mode=SearchMode(mode), subreddit=subreddit)

        rows, depth = retrieve_threads(search, "fact", "keyword")
        self.assertEqual(len(rows), 10)
        self.assertEqual(depth, 200)
        self.assertEqual([call.kwargs["limit"] for call in store.fts_search.call_args_list], [50, 100, 200])
        store.fts_search.reset_mock()
        store.fts_search.side_effect = lambda query, limit, subreddit: [
            {"id": f"p{i}", "content_type": "post"} for i in range(limit)
        ]
        rows, depth = retrieve_threads(search, "fact", "keyword")
        self.assertEqual(len(rows), 10)
        self.assertEqual(depth, 50)
        self.assertEqual([call.kwargs["limit"] for call in store.fts_search.call_args_list], [50])

    @unittest.skipUnless(importlib.util.find_spec("lancedb"), "index extra not installed")
    def test_external_evaluation_measures_public_thread_search_without_raw_overfetch(self):
        from rsi.evaluation import export_fixture

        depths = []
        engine = MagicMock()
        engine.store.count.return_value = 120
        engine.search_candidates.side_effect = AssertionError("evaluation must use public source-thread search")

        def candidates(query, *, limit, mode, subreddit):
            depths.append(limit)
            return [{"id": f"chunk{i}", "content_type": "post", "parent_doc_id": f"post:f{i + 1:03}"}
                    for i in range(min(limit, 10))]

        engine.search.side_effect = candidates
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_fixture(root / "fixture")
            (root / "index").mkdir()
            args = ["eval", "--dataset", str(root / "fixture/questions.jsonl"), "--qrels",
                    str(root / "fixture/qrels.jsonl"), "--corpus", str(root / "fixture/corpus"),
                    "--db-path", str(root / "index"), "--output", str(root / "run.json")]
            with patch("rsi.indexer.search.SearchEngine", return_value=engine), patch(
                "rsi.evaluation.index_fingerprint", return_value={"source_thread_count": 10}
            ):
                result = CliRunner().invoke(app, args, env={"RSI_EMBEDDING_PROVIDER": "local"})
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            report = json.loads((root / "run.json").read_text())
        self.assertTrue(depths and all(depth == 10 for depth in depths))
        self.assertTrue(all(row["candidate_depth"] is None for row in report["queries"]))
        self.assertEqual(report["retrieval_unit"], "source_threads")
        engine.search_candidates.assert_not_called()

    def test_fixture_rejects_semantic_and_missing_explicit_inputs(self):
        runner = CliRunner()
        for args in [["eval", "--fixture", "--mode", "semantic"], ["eval"]]:
            with self.subTest(args=args):
                result = runner.invoke(app, args)
                self.assertNotEqual(result.exit_code, 0)
                self.assertNotIn("No such command", result.output)

    @unittest.skipUnless(importlib.util.find_spec("lancedb"), "index extra not installed")
    def test_existing_empty_or_unrelated_index_fails_instead_of_scoring_zero(self):
        from rsi.evaluation import export_fixture
        from rsi.indexer.vector_store import VectorStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            export_fixture(root / "fixture")
            db = root / "index"
            db.mkdir()
            args = ["eval", "--dataset", str(root / "fixture/questions.jsonl"), "--qrels",
                    str(root / "fixture/qrels.jsonl"), "--corpus", str(root / "fixture/corpus"),
                    "--db-path", str(db), "--output", str(root / "run.json")]
            env = {"RSI_EMBEDDING_PROVIDER": "local", "RSI_EMBEDDING_MODEL": "BAAI/bge-m3"}
            result = CliRunner().invoke(app, args, env=env)
            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertIn("Evaluation failed:", result.output)
            self.assertFalse((root / "run.json").exists())
            metadata = db / "rsi_index.json"
            metadata.write_text("[]")
            result = CliRunner().invoke(app, args, env=env)
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Evaluation failed:", result.output)
            metadata.write_text('{"provider":"local","model":"BAAI/bge-m3","complete":true}')
            store = VectorStore(db)
            store.add_records([{"id": "unrelated", "content_type": "post", "text": "other corpus",
                                "subreddit": "other"}])
            store.create_fts_index()
            result = CliRunner().invoke(app, args, env=env)
            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertIn("Evaluation failed:", result.output)

    def test_hosted_embedding_requires_opt_in_and_names_destination(self):
        args = ["eval", "--mode", "semantic", "--dataset", "/missing/questions.jsonl",
                "--qrels", "/missing/qrels.jsonl", "--corpus", "/missing/corpus", "--db-path", "/missing/index"]
        env = {"RSI_EMBEDDING_PROVIDER": "openai", "RSI_EMBEDDING_BASE_URL": "https://host.example/v1"}
        result = CliRunner().invoke(app, args, env=env, color=True)
        self.assertEqual(result.exit_code, 2)
        output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        self.assertIn("--allow-remote", output)
        self.assertIn("https://host.example/v1", output)
        result = CliRunner().invoke(app, args + ["--allow-remote"], env=env)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Embedding destination: https://host.example/v1", result.output)

    @unittest.skipUnless(importlib.util.find_spec("lancedb"), "index extra not installed")
    def test_public_fixture_has_fifty_answerable_and_ten_absent_without_embedding(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "rsi.indexer.embedder.make_embedder", side_effect=AssertionError("embedding must stay offline")
        ):
            output = Path(tmp) / "run.json"
            result = CliRunner().invoke(app, ["eval", "--fixture", "--split", "all", "--output", str(output)],
                                        env={"RSI_EMBEDDING_PROVIDER": "openai"})
            self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
            report = json.loads(output.read_text())
            self.assertEqual(report["answerable_count"], 50)
            self.assertEqual(report["absent_count"], 10)
            self.assertEqual(report["error_count"], 0)
            self.assertEqual(report["identity"]["embedding"], "none (keyword)")
            self.assertEqual(report["absent_answers"]["unsupported_answer_count"], None)
            self.assertEqual(len(report["queries"]), 60)
            self.assertTrue(all(row["metrics"] is None for row in report["queries"] if not row["answerable"]))
            self.assertTrue(all(row["ranking"] for row in report["queries"] if row["answerable"]))


if __name__ == "__main__":
    unittest.main()
