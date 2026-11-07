"""CLI evaluation runs locally and errors cannot become successful scores."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from rsi.cli import app


class TestEvalCommand(unittest.TestCase):
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
        result = CliRunner().invoke(app, args, env=env)
        self.assertEqual(result.exit_code, 2)
        self.assertIn("--allow-remote", result.output)
        self.assertIn("https://host.example/v1", result.output)
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
