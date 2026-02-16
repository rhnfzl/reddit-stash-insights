"""RAG Quality Test: evaluate Direct RAG answer quality.

Runs test questions through the Direct RAG engine (SearchEngine + prompt + LLM)
multiple times, evaluates answer quality using RAGAS-inspired metrics (automated,
no judge LLM needed), and produces a quality report.

Test questions are loaded from tests/fixtures/rag_test_questions.json for reuse
across regression tests, prompt tuning, and model swaps.

Usage:
    python ab_test_rag.py --model /path/to/model.gguf
    python ab_test_rag.py --model /path/to/model.gguf --questions 5  # quick subset
    python ab_test_rag.py --model /path/to/model.gguf --category fitness  # filter
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterator

from rsi.config import DEFAULT_DB_PATH
from rsi.indexer.search import SearchEngine, SearchMode

# ──────────────────────────────────────────────────────────
# Test Questions — loaded from fixture file
# ──────────────────────────────────────────────────────────

FIXTURE_PATH = Path(__file__).parent / "tests" / "fixtures" / "rag_test_questions.json"


def load_test_questions(path: Path = FIXTURE_PATH) -> list[dict]:
    """Load test questions from the JSON fixture file."""
    with open(path) as f:
        data = json.load(f)
    return data["questions"]


# ──────────────────────────────────────────────────────────
# LLM Provider (llama-cpp)
# ──────────────────────────────────────────────────────────

class LlamaCppProvider:
    """Minimal llama-cpp-python provider for RAG quality testing."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,  # -1 = offload all to GPU (Metal)
            verbose=False,
        )

    def generate(self, messages: list[dict]) -> str:
        response = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        return response["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict]) -> Iterator[str]:
        for chunk in self._llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.3,
            stream=True,
        ):
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                yield delta["content"]


# ──────────────────────────────────────────────────────────
# RAGAS-Inspired Evaluation Metrics (automated, no judge LLM)
# ──────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should can could may might must need dare "
    "i me my we our you your he she it they them their "
    "this that these those which what who whom whose "
    "and but or nor not so yet for in on at to from by with about "
    "of as if then than how when where there here all any each "
    "few more most no some such only same just also very".split()
)


def _content_words(text: str) -> set[str]:
    """Extract meaningful content words (lowercase, no stop words, len >= 3)."""
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return {w for w in words if w not in _STOP_WORDS and len(w) >= 3}


def score_topic_hits(answer: str, expected_topics: list[str]) -> float:
    """Topic Hit Rate: fraction of expected topics found in the answer."""
    answer_lower = answer.lower()
    hits = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
    return round(hits / len(expected_topics), 3) if expected_topics else 0.0


def score_citation_accuracy(answer: str, num_sources: int) -> float:
    """Citation Accuracy: fraction of [N] citations that reference valid sources.

    Checks that cited reference numbers [1], [2], etc. are within the range
    of actually retrieved sources. High score = citations are properly anchored.
    """
    citations = re.findall(r"\[(\d+)\]", answer)
    if not citations:
        return 0.0
    valid = sum(1 for c in citations if 1 <= int(c) <= num_sources)
    return round(valid / len(citations), 3)


def score_citation_density(answer: str) -> float:
    """Citation Density: how many unique source citations appear in the answer.

    Normalized to 0-1 range (caps at 5 unique citations = 1.0).
    """
    unique_citations = set(re.findall(r"\[(\d+)\]", answer))
    return round(min(len(unique_citations) / 5.0, 1.0), 3)


def score_context_relevance(query: str, docs: list[dict]) -> float:
    """Context Relevance (RAGAS-inspired Context Precision).

    Measures what fraction of retrieved documents contain query-relevant terms.
    Higher = retrieval is returning relevant results, not noise.
    """
    if not docs:
        return 0.0
    query_words = _content_words(query)
    relevant = 0
    for doc in docs:
        doc_text = doc.get("text", "").lower()
        overlap = sum(1 for w in query_words if w in doc_text)
        if overlap >= max(1, len(query_words) * 0.3):  # at least 30% of query terms
            relevant += 1
    return round(relevant / len(docs), 3)


def score_faithfulness(answer: str, context: str) -> float:
    """Faithfulness Proxy (RAGAS-inspired Faithfulness).

    Measures what fraction of content words in the answer also appear in the
    retrieved context. Higher = answer draws from context rather than hallucinating.
    A perfect score means every substantive word in the answer can be traced to context.
    """
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_lower = context.lower()
    grounded = sum(1 for w in answer_words if w in context_lower)
    return round(grounded / len(answer_words), 3)


def score_groundedness(answer: str, context: str) -> float:
    """Groundedness: fraction of answer sentences with supporting evidence in context.

    Splits answer into sentences, checks each sentence for content word overlap
    with the context. Sentences with >= 40% overlap are considered grounded.
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if len(s.strip()) > 10]
    if not sentences:
        return 0.0
    context_words = _content_words(context)
    grounded = 0
    for sent in sentences:
        sent_words = _content_words(sent)
        if not sent_words:
            continue
        overlap = len(sent_words & context_words) / len(sent_words)
        if overlap >= 0.4:
            grounded += 1
    return round(grounded / len(sentences), 3) if sentences else 0.0


def score_answer_completeness(answer: str) -> float:
    """Answer Completeness: heuristic score for answer quality.

    Penalizes refusals, very short answers, and rewards structured responses.
    """
    if not answer or len(answer.strip()) < 10:
        return 0.0

    score = 0.0
    answer_lower = answer.lower()

    # Refusal detection (penalize)
    refusal_phrases = [
        "i don't have enough", "the context doesn't", "no relevant information",
        "cannot answer", "not enough context", "i'm unable to",
    ]
    is_refusal = any(phrase in answer_lower for phrase in refusal_phrases)
    if is_refusal:
        return 0.1  # not zero — correctly refusing is better than hallucinating

    # Length scoring (diminishing returns after ~200 chars)
    length = len(answer.strip())
    if length >= 200:
        score += 0.3
    elif length >= 100:
        score += 0.2
    elif length >= 50:
        score += 0.1

    # Has citations
    if re.search(r"\[\d+\]", answer):
        score += 0.3

    # Has structure (bullet points, numbered lists, paragraphs)
    if re.search(r"[\n-].*[\n-]", answer) or re.search(r"\d\.", answer):
        score += 0.2

    # Contains specific details (numbers, proper nouns, etc.)
    specific_patterns = re.findall(r"\b[A-Z][a-z]+\b|\b\d+\b", answer)
    if len(specific_patterns) >= 3:
        score += 0.2

    return round(min(score, 1.0), 3)


def compute_composite_score(metrics: dict[str, float]) -> float:
    """Weighted composite quality score from all metrics.

    Weights reflect importance for RAG quality:
    - Faithfulness (0.25): most critical — don't hallucinate
    - Topic Hit Rate (0.20): answers the actual question
    - Groundedness (0.15): stays within context
    - Context Relevance (0.15): retrieval quality
    - Citation Accuracy (0.10): proper source attribution
    - Answer Completeness (0.10): structural quality
    - Citation Density (0.05): breadth of source usage
    """
    weights = {
        "faithfulness": 0.25,
        "topic_hit_rate": 0.20,
        "groundedness": 0.15,
        "context_relevance": 0.15,
        "citation_accuracy": 0.10,
        "answer_completeness": 0.10,
        "citation_density": 0.05,
    }
    total = sum(metrics.get(k, 0.0) * w for k, w in weights.items())
    return round(total, 3)


# ──────────────────────────────────────────────────────────
# Direct RAG Engine
# ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the user's saved Reddit content. Use ONLY the provided context to answer. If the context doesn't contain relevant information, say so. Always cite sources using their reference number [1], [2], etc. Be concise."""


def build_context_block(docs: list[dict]) -> str:
    """Format retrieved docs as numbered context."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        sub = doc.get("subreddit", "?")
        score = doc.get("score", 0)
        ctype = doc.get("content_type", "?")
        text = doc.get("text", "")[:500]
        blocks.append(f"[{i}] r/{sub} | {ctype} | Score: {score}\n{text}")
    return "\n\n".join(blocks)


def direct_rag(query: str, search_engine: SearchEngine, llm: LlamaCppProvider, limit: int = 5) -> dict:
    """Direct RAG: our SearchEngine + prompt + LLM."""
    start = time.time()

    # Retrieve
    docs = search_engine.search(query, limit=limit, mode=SearchMode.HYBRID)
    retrieval_time = time.time() - start

    # Build prompt
    context = build_context_block(docs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {query}"},
    ]

    # Generate
    gen_start = time.time()
    answer = llm.generate(messages)
    gen_time = time.time() - gen_start

    total_time = time.time() - start

    return {
        "engine": "direct",
        "answer": answer,
        "context": context,
        "docs": docs,
        "sources": [{"subreddit": d.get("subreddit"), "id": d.get("id"), "content_type": d.get("content_type")} for d in docs],
        "num_sources": len(docs),
        "retrieval_time_s": round(retrieval_time, 3),
        "generation_time_s": round(gen_time, 3),
        "total_time_s": round(total_time, 3),
    }


# ──────────────────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    question_id: str
    question: str
    category: str
    expected_topics: list[str]
    engine: str
    run: int
    answer: str
    sources: list[dict]
    num_sources: int
    retrieval_time_s: float | None
    generation_time_s: float | None
    total_time_s: float
    # RAGAS-inspired metrics
    topic_hit_rate: float = 0.0
    citation_accuracy: float = 0.0
    citation_density: float = 0.0
    context_relevance: float = 0.0
    faithfulness: float = 0.0
    groundedness: float = 0.0
    answer_completeness: float = 0.0
    composite_score: float = 0.0


def evaluate_result(result: dict, question: dict) -> dict[str, float]:
    """Compute all RAGAS-inspired metrics for a single RAG result."""
    answer = result["answer"]
    context = result.get("context", "")
    query = question["question"]
    expected = question["expected_topics"]
    docs = result.get("docs", [])
    num_sources = result["num_sources"]

    metrics = {
        "topic_hit_rate": score_topic_hits(answer, expected),
        "citation_accuracy": score_citation_accuracy(answer, num_sources),
        "citation_density": score_citation_density(answer),
        "context_relevance": score_context_relevance(query, docs),
        "faithfulness": score_faithfulness(answer, context),
        "groundedness": score_groundedness(answer, context),
        "answer_completeness": score_answer_completeness(answer),
    }
    metrics["composite_score"] = compute_composite_score(metrics)
    return metrics


def run_quality_test(
    model_path: str,
    db_path: Path = DEFAULT_DB_PATH,
    num_questions: int | None = None,
    category: str | None = None,
    runs_per_question: int = 3,
) -> list[TestResult]:
    """Run quality test: each question x N runs through Direct RAG."""

    questions = load_test_questions()

    if category:
        questions = [q for q in questions if q["category"] == category]
    if num_questions:
        questions = questions[:num_questions]

    print(f"Test config: {len(questions)} questions x {runs_per_question} runs")

    print(f"Loading llama-cpp model: {model_path}")
    llm = LlamaCppProvider(model_path)

    print(f"Loading search engine from: {db_path}")
    search_engine = SearchEngine(db_path=db_path)

    results: list[TestResult] = []
    total = len(questions) * runs_per_question
    current = 0

    for q in questions:
        for run_num in range(1, runs_per_question + 1):
            current += 1
            print(f"\n[{current}/{total}] Run {run_num} | {q['id']}: {q['question'][:60]}...")

            try:
                result = direct_rag(q["question"], search_engine, llm)
                metrics = evaluate_result(result, q)

                tr = TestResult(
                    question_id=q["id"],
                    question=q["question"],
                    category=q["category"],
                    expected_topics=q["expected_topics"],
                    engine="direct",
                    run=run_num,
                    answer=result["answer"],
                    sources=result["sources"],
                    num_sources=result["num_sources"],
                    retrieval_time_s=result["retrieval_time_s"],
                    generation_time_s=result["generation_time_s"],
                    total_time_s=result["total_time_s"],
                    **metrics,
                )
                results.append(tr)

                print(f"  Score: {metrics['composite_score']:.0%} | Topics: {metrics['topic_hit_rate']:.0%} | Faith: {metrics['faithfulness']:.0%} | Time: {result['total_time_s']}s")
                print(f"  Answer: {result['answer'][:150]}...")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append(TestResult(
                    question_id=q["id"],
                    question=q["question"],
                    category=q["category"],
                    expected_topics=q["expected_topics"],
                    engine="direct",
                    run=run_num,
                    answer=f"ERROR: {e}",
                    sources=[],
                    num_sources=0,
                    retrieval_time_s=None,
                    generation_time_s=None,
                    total_time_s=0,
                ))

    return results


# ──────────────────────────────────────────────────────────
# Summary Report
# ──────────────────────────────────────────────────────────

METRIC_NAMES = [
    "composite_score", "topic_hit_rate", "faithfulness", "groundedness",
    "context_relevance", "citation_accuracy", "citation_density", "answer_completeness",
]


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_summary(results: list[TestResult]) -> None:
    """Print quality report with all metrics."""
    valid = [r for r in results if not r.answer.startswith("ERROR")]
    errors = len(results) - len(valid)

    print("\n" + "=" * 70)
    print("RAG QUALITY REPORT — RAGAS-Inspired Evaluation")
    print("=" * 70)

    print(f"\n  Runs: {len(results)}  |  Errors: {errors}  |  Avg time: {_avg([r.total_time_s for r in results]):.2f}s")
    gen_times = [r.generation_time_s for r in valid if r.generation_time_s]
    ret_times = [r.retrieval_time_s for r in valid if r.retrieval_time_s]
    if gen_times:
        print(f"  Avg generation: {_avg(gen_times):.2f}s  |  Avg retrieval: {_avg(ret_times):.3f}s")

    # ── Metric overview ──
    print(f"\n  {'Metric':<25s} {'Mean':>6s}  {'StdDev':>7s}  {'Min':>5s}  {'Max':>5s}")
    print(f"  {'─' * 50}")
    for metric in METRIC_NAMES:
        values = [getattr(r, metric) for r in valid]
        if values:
            mean = _avg(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"  {metric:<25s} {mean:>5.0%}  {std:>6.1%}  {min(values):>4.0%}  {max(values):>4.0%}")

    # ── Per-category breakdown ──
    print(f"\n{'━' * 45}")
    print("  PER-CATEGORY COMPOSITE SCORES")
    print(f"{'━' * 45}")
    categories = sorted(set(r.category for r in results))
    print(f"  {'Category':<15s} {'Score':>8s}")
    print(f"  {'─' * 25}")
    for cat in categories:
        values = [r.composite_score for r in valid if r.category == cat]
        print(f"  {cat:<15s} {_avg(values):>7.0%}")

    # ── Consistency analysis ──
    print(f"\n{'━' * 60}")
    print("  CONSISTENCY (variance across repeated runs)")
    print(f"{'━' * 60}")
    question_ids = sorted(set(r.question_id for r in valid))
    consistent = 0
    total_q = 0
    for qid in question_ids:
        scores = [r.composite_score for r in valid if r.question_id == qid]
        if len(scores) >= 2:
            total_q += 1
            spread = max(scores) - min(scores)
            if spread <= 0.15:
                consistent += 1
                status = "stable"
            elif spread <= 0.30:
                status = f"moderate ({min(scores):.0%}-{max(scores):.0%})"
            else:
                status = f"UNSTABLE ({min(scores):.0%}-{max(scores):.0%})"
            print(f"    {qid:10s}  scores: {[f'{s:.0%}' for s in scores]}  {status}")
    if total_q:
        print(f"    Stability rate: {consistent}/{total_q} ({consistent/total_q:.0%})")

    # ── Worst performers (for tuning) ──
    print(f"\n{'━' * 60}")
    print("  BOTTOM 5 QUESTIONS (lowest composite, for tuning)")
    print(f"{'━' * 60}")
    qid_scores = {}
    for r in valid:
        qid_scores.setdefault(r.question_id, []).append(r.composite_score)
    qid_avg = {qid: _avg(scores) for qid, scores in qid_scores.items()}
    for qid, avg_score in sorted(qid_avg.items(), key=lambda x: x[1])[:5]:
        q_text = next((r.question for r in results if r.question_id == qid), "")
        print(f"    {qid:10s} {avg_score:>4.0%}  {q_text[:60]}")


def save_results(results: list[TestResult], output_path: Path) -> None:
    """Save full results to JSON for later analysis and regression comparison."""
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(results),
            "engine": "direct",
            "questions": len(set(r.question_id for r in results)),
            "fixture_file": str(FIXTURE_PATH),
        },
        "results": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nFull results saved to: {output_path}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG quality test for Direct RAG engine")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="LanceDB path")
    parser.add_argument("--questions", type=int, default=None, help="Limit number of questions (default: all)")
    parser.add_argument("--category", default=None, help="Filter by question category (e.g. fitness, tech)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per question")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    results = run_quality_test(
        model_path=args.model,
        db_path=Path(args.db_path),
        num_questions=args.questions,
        category=args.category,
        runs_per_question=args.runs,
    )

    print_summary(results)

    output_path = Path(args.output) if args.output else Path(f"ab_test_results_{datetime.now():%Y%m%d_%H%M%S}.json")
    save_results(results, output_path)
