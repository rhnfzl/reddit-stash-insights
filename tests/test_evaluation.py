"""Retrieval scores use independent hand-calculated relevance labels."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from rsi.evaluation import canonical_doc_id, evaluate, load_dataset, retrieve_threads, score_ranking, snapshot_manifest


class TestScores(unittest.TestCase):
    def test_perfect_missed_reversed_partial_and_empty(self):
        labels = {"post:a": 2, "post:b": 1, "post:c": 0}
        ideal = 3 + 1 / math.log2(3)
        cases = [
            (["post:a", "post:b"], 1.0, 1.0, 1.0),
            (["post:c"], 0.0, 0.0, 0.0),
            (["post:b", "post:a"], 1.0, 1.0, (1 + 3 / math.log2(3)) / ideal),
            (["post:c", "post:b"], 0.5, 0.5, (1 / math.log2(3)) / ideal),
            ([], 0.0, 0.0, 0.0),
        ]
        for ranking, recall, mrr, ndcg in cases:
            with self.subTest(ranking=ranking):
                result = score_ranking(ranking, labels)
                self.assertAlmostEqual(result["recall@10"], recall)
                self.assertAlmostEqual(result["mrr@10"], mrr)
                self.assertAlmostEqual(result["ndcg@10"], ndcg)

    def test_duplicate_threads_do_not_consume_rank_or_gain(self):
        scores = score_ranking(["post:a", "post:a", "post:b"], {"post:a": 1, "post:b": 2})
        self.assertAlmostEqual(scores["ndcg@10"], 0.7967075809905066)
        self.assertEqual(scores["recall@10"], 1.0)

    def test_bad_rankings_and_labels_are_errors(self):
        cases: list[tuple[list[str], dict[str, Any]]] = [
            (["chunk:a"], {"post:a": 2}), (["post:a"], {"post:a": 3}),
            (["post:a"], {"post:a": True}), ([], {"post:a": 0}),
        ]
        for ranking, labels in cases:
            with self.subTest(ranking=ranking, labels=labels), self.assertRaises(ValueError):
                score_ranking(ranking, labels)

    def test_cutoff_counts_only_first_ten_unique_documents(self):
        result = score_ranking([f"post:{n}" for n in range(11)], {"post:10": 2})
        self.assertEqual(result["recall@10"], 0.0)
        self.assertEqual(result["mrr@10"], 0.0)


class TestThreadIdentity(unittest.TestCase):
    def test_permalink_comment_and_chunk_share_source_thread(self):
        rows = [
            {"id": "p", "content_type": "post"},
            {"id": "c", "content_type": "comment", "permalink": "https://reddit.com/r/x/comments/p/title/c/"},
            {"id": "chunk", "content_type": "post", "parent_doc_id": "post:p"},
        ]
        self.assertEqual([canonical_doc_id(row) for row in rows], ["post:p"] * 3)
        self.assertEqual(canonical_doc_id({"id": "c", "content_type": "comment"}), "comment:c")

    def test_thread_permalink_accepts_terminal_id_and_url_query(self):
        for permalink in ("/r/x/comments/p", "https://reddit.com/comments/p?context=1"):
            with self.subTest(permalink=permalink):
                self.assertEqual(canonical_doc_id({"id": "c", "content_type": "comment", "permalink": permalink}), "post:p")

    def test_overfetch_fills_unique_threads_and_records_depth(self):
        calls = []
        def search(query, limit, **kwargs):
            calls.append(limit)
            return [{"id": f"chunk{i}", "content_type": "post", "parent_doc_id": f"post:{i // 12}"}
                    for i in range(min(limit, 120))]
        rows, depth = retrieve_threads(search, "fact", "keyword")
        self.assertEqual(calls, [50, 100, 200])
        self.assertEqual(depth, 200)
        self.assertEqual([canonical_doc_id(row) for row in rows], [f"post:{n}" for n in range(10)])

    def test_exhaustion_returns_fewer_than_ten(self):
        rows, depth = retrieve_threads(lambda *a, **kw: [{"id": "a", "content_type": "post"}] * 3,
                                       "fact", "keyword")
        self.assertEqual(len(rows), 1)
        self.assertEqual(depth, 50)


class TestDatasetValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.corpus = self.root / "corpus"
        (self.corpus / "x").mkdir(parents=True)
        (self.corpus / "x" / "POST_a.md").write_text("---\nid: a\nsubreddit: x\n---\n# Fact\nAn orchid needs rainwater.")
        self.question = {"query_id": "q1", "query": "orchid water", "split": "dev", "tags": ["exact"],
                         "answerable": True, "reference_answer": "Rainwater", "evidence": ["post:a"]}
        self.qrel = {"query_id": "q1", "doc_id": "post:a", "grade": 2}
        self.write()

    def write(self):
        (self.root / "questions.jsonl").write_text(json.dumps(self.question) + "\n")
        (self.root / "qrels.jsonl").write_text(json.dumps(self.qrel) + "\n")
        (self.root / "snapshot.json").write_text(json.dumps(snapshot_manifest(self.corpus)))

    def load(self):
        return load_dataset(self.root / "questions.jsonl", self.root / "qrels.jsonl", self.corpus)

    def test_snapshot_is_verified_before_evaluation(self):
        self.load()
        (self.corpus / "x" / "POST_a.md").write_text("changed")
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.load()

    def test_malformed_manifest_and_unhashable_label_ids_raise_validation_errors(self):
        value: Any
        for value in ([], {"files": [], "sha256": "x"}, {"documents": [[]]}):
            self.write()
            (self.root / "snapshot.json").write_text(json.dumps(value))
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load()
        for key in ("query_id", "doc_id"):
            self.qrel = {"query_id": "q1", "doc_id": "post:a", "grade": 2, key: []}
            self.write()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load()
        self.qrel = {"query_id": "q1", "doc_id": "post:a", "grade": 2}
        self.question["evidence"] = [{"doc_id": []}]
        self.write()
        with self.assertRaises(ValueError):
            self.load()

    def test_evidence_quote_must_exist_in_the_frozen_source(self):
        self.question["evidence"] = [{"doc_id": "post:a", "quote": "Invented evidence"}]
        self.write()
        with self.assertRaisesRegex(ValueError, "evidence"):
            self.load()

    def test_duplicate_queries_invalid_grades_and_missing_documents_fail(self):
        for field, value in [("doc_id", "post:missing"), ("grade", 3), ("query_id", "missing")]:
            with self.subTest(field=field):
                self.qrel = {"query_id": "q1", "doc_id": "post:a", "grade": 2, field: value}
                self.write()
                with self.assertRaises(ValueError):
                    self.load()
        self.qrel = {"query_id": "q1", "doc_id": "post:a", "grade": 2}
        self.write()
        path = self.root / "questions.jsonl"
        path.write_text(path.read_text() * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.load()


if __name__ == "__main__":
    unittest.main()


class TestReports(unittest.TestCase):
    def test_public_thread_results_preserve_first_alias_and_requested_cutoff(self):
        snapshot = {"sha256": "fixed", "documents": ["post:a", "post:b"],
                    "aliases": {"comment:c": "post:a"}}
        calls = []

        def search(query, **kwargs):
            calls.append(kwargs)
            return [{"id": "a", "content_type": "post"}, {"id": "b", "content_type": "post"},
                    {"id": "c", "content_type": "comment"}]

        report = evaluate([{"query_id": "q", "query": "fact", "answerable": True, "split": "dev", "tags": []}],
                          {"q": {"post:a": 2}}, snapshot, search, mode="keyword", split="dev", identity={},
                          thread_search=True)
        self.assertEqual(calls, [{"limit": 10, "mode": "keyword", "subreddit": None}])
        self.assertEqual([r["doc_id"] for r in report["queries"][0]["ranking"]], ["post:a", "post:b"])
        self.assertEqual(report["queries"][0]["ranking"][0]["evidence_id"], "post:a")
        self.assertIsNone(report["queries"][0]["candidate_depth"])

    def setUp(self):
        self.queries = [{"query_id": "a", "query": "water", "answerable": True, "split": "dev", "tags": []},
                        {"query_id": "b", "query": "missing", "answerable": False, "split": "dev", "tags": []}]
        self.labels = {"a": {"post:p": 2}, "b": {}}
        self.snapshot: dict[str, Any] = {
            "sha256": "test", "documents": ["post:p"], "aliases": {"comment:c": "post:p"},
        }

    def run_eval(self, search):
        return evaluate(self.queries, self.labels, self.snapshot, search, mode="keyword", split="dev", identity={})

    def test_comment_aliases_and_absent_denominators_are_preserved(self):
        report = self.run_eval(lambda *a, **kw: [{"id": "c", "content_type": "comment"}])
        self.assertEqual(report["metrics"], {"recall@10": 1.0, "mrr@10": 1.0, "ndcg@10": 1.0})
        self.assertEqual(report["answerable_count"], 1)
        self.assertEqual(report["absent_count"], 1)
        self.assertIsNone(report["queries"][1]["metrics"])
        self.assertEqual(report["queries"][0]["ranking"][0]["evidence_id"], "comment:c")

    def test_retrieval_errors_invalidate_aggregate_scores(self):
        def broken(*a, **kw):
            raise RuntimeError("search failed")
        report = self.run_eval(broken)
        self.assertEqual(report["error_count"], 2)
        self.assertIsNone(report["metrics"])
        self.assertIn("RuntimeError", report["queries"][0]["error"])

    def test_unknown_source_is_error_but_unjudged_known_source_is_unknown(self):
        report = self.run_eval(lambda *a, **kw: [{"id": "outside", "content_type": "post"}])
        self.assertEqual(report["error_count"], 2)
        self.snapshot["documents"].append("post:u")
        report = self.run_eval(lambda *a, **kw: [{"id": "u", "content_type": "post"}])
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["judged_coverage"], 0.0)
        self.assertIsNone(report["queries"][0]["ranking"][0]["grade"])

    def test_report_latency_percentiles_and_rss_have_units(self):
        report = self.run_eval(lambda *a, **kw: [{"id": "p", "content_type": "post"}])
        self.assertGreaterEqual(report["latency_p95_ms"], report["latency_p50_ms"])
        if report["peak_rss_bytes"] is not None:
            self.assertGreater(report["peak_rss_bytes"], 0)

    def test_rankings_and_scores_are_deterministic(self):
        def search(*a, **kw):
            return [{"id": "p", "content_type": "post"}]
        first, second = self.run_eval(search), self.run_eval(search)
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual([q["ranking"] for q in first["queries"]], [q["ranking"] for q in second["queries"]])
