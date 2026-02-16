"""Nox sessions for multi-version Python compatibility testing.

Usage:
    nox                     # run all sessions
    nox -s core             # core tests only (fast, no ML deps)
    nox -s search           # search tests (requires ML deps)
    nox -s lint             # lint only
    nox -l                  # list available sessions

Requires Python 3.11-3.14 installed locally (e.g. via pyenv or uv).
"""

import nox

PYTHON_VERSIONS = ["3.11", "3.12", "3.13", "3.14"]
CORE_TESTS = [
    "tests.test_models",
    "tests.test_parser",
    "tests.test_scanner",
    "tests.test_cli",
    "tests.test_config",
    "tests.test_s3_fetch",
]
SEARCH_TESTS = [
    "tests.test_embedder",
    "tests.test_vector_store",
    "tests.test_search",
    "tests.test_cli_search",
]


@nox.session(python=PYTHON_VERSIONS)
def core(session: nox.Session) -> None:
    """Run core tests (no ML dependencies) across Python versions."""
    session.install("-e", ".[dev]")
    session.run("python", "-m", "unittest", *CORE_TESTS, "-v")


@nox.session(python=["3.11", "3.12", "3.13"])
def search(session: nox.Session) -> None:
    """Run search tests (requires sentence-transformers, lancedb, FlagEmbedding).

    Python 3.14 excluded: ML dependencies (torch, etc.) may not have wheels yet.
    """
    session.install("-e", ".[search,dev]")
    session.run("python", "-m", "unittest", *SEARCH_TESTS, "-v")


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    """Run ruff linter across Python versions."""
    session.install("ruff>=0.15")
    session.run("ruff", "check", "src/", "tests/")
