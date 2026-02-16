"""Tests for CLI chat command and chat-related config."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner


class TestChatConfigDefaults(unittest.TestCase):
    """Test chat-specific settings in config.py."""

    def test_default_chat_settings(self):
        from rsi.config import Settings

        s = Settings()
        self.assertEqual(s.chat_context_docs, 5)
        self.assertEqual(s.chat_search_mode, "hybrid")
        self.assertEqual(s.chat_max_history, 10)

    def test_chat_settings_from_toml(self):
        from rsi.config import Settings

        toml_content = (
            b'[chat]\ncontext_docs = 8\nsearch_mode = "semantic"\nmax_history = 20\n'
        )
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            s = Settings.load(config_path=config_path)
            self.assertEqual(s.chat_context_docs, 8)
            self.assertEqual(s.chat_search_mode, "semantic")
            self.assertEqual(s.chat_max_history, 20)
        finally:
            config_path.unlink()

    def test_chat_env_var_overrides(self):
        from rsi.config import Settings

        toml_content = b'[chat]\ncontext_docs = 8\nsearch_mode = "semantic"\n'
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            config_path = Path(f.name)

        try:
            os.environ["RSI_CHAT_CONTEXT_DOCS"] = "12"
            os.environ["RSI_CHAT_SEARCH_MODE"] = "keyword"
            s = Settings.load(config_path=config_path)
            self.assertEqual(s.chat_context_docs, 12)
            self.assertEqual(s.chat_search_mode, "keyword")
        finally:
            config_path.unlink()
            del os.environ["RSI_CHAT_CONTEXT_DOCS"]
            del os.environ["RSI_CHAT_SEARCH_MODE"]


class TestChatCommandSingleTurn(unittest.TestCase):
    """Test `rsi chat "question"` single-turn mode."""

    @patch("rsi.cli._build_chat_engine")
    def test_single_turn_prints_answer_and_sources(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.chat.return_value = MagicMock(
            answer="The answer is 42.",
            sources=[
                {"subreddit": "Python", "text": "some context", "file_path": "/test.md"},
            ],
        )
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "What is the answer?", "--no-stream"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("The answer is 42.", result.output)
        self.assertIn("r/Python", result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_single_turn_streaming(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.chat_stream.return_value = (
            iter(["The ", "answer ", "is ", "42."]),
            [{"subreddit": "Python", "text": "ctx", "file_path": "/t.md"}],
        )
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "What is the answer?"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        # Streaming output should contain the full answer
        self.assertIn("The answer is 42.", result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_single_turn_no_results(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.chat.return_value = MagicMock(
            answer="I don't have relevant information.",
            sources=[],
        )
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "obscure question", "--no-stream"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("I don't have relevant information.", result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_cli_options_forwarded(self, mock_build):
        """CLI --mode, --limit options are respected."""
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.chat.return_value = MagicMock(answer="ok", sources=[])
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, [
            "chat", "test", "--no-stream", "--mode", "semantic", "--limit", "3",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_engine.chat.assert_called_once_with("test", limit=3)

    @patch("rsi.cli._build_chat_engine")
    def test_provider_and_model_options(self, mock_build):
        """CLI --provider and --model are passed to engine builder."""
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.chat.return_value = MagicMock(answer="ok", sources=[])
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, [
            "chat", "test", "--no-stream",
            "--provider", "openai", "--model", "gpt-4o",
        ])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args
        # _build_chat_engine(provider, model, db_path, mode, max_history, context_docs)
        self.assertEqual(call_kwargs[1]["provider"], "openai")
        self.assertEqual(call_kwargs[1]["model"], "gpt-4o")


class TestChatCommandREPL(unittest.TestCase):
    """Test `rsi chat` interactive REPL mode."""

    @patch("rsi.cli._build_chat_engine")
    def test_repl_quit_command(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="/quit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_repl_help_command(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="/help\n/quit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("/quit", result.output)
        self.assertIn("/clear", result.output)
        self.assertIn("/sources", result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_repl_clear_command(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="/clear\n/quit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_engine.clear_history.assert_called_once()
        self.assertIn("cleared", result.output.lower())

    @patch("rsi.cli._build_chat_engine")
    def test_repl_sources_command_no_history(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.history = MagicMock()
        mock_engine.history.last_sources.return_value = []
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="/sources\n/quit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("no sources", result.output.lower())

    @patch("rsi.cli._build_chat_engine")
    def test_repl_sources_command_with_history(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.history = MagicMock()
        mock_engine.history.last_sources.return_value = [
            {"subreddit": "Python", "text": "context text", "file_path": "/a.md"},
        ]
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="/sources\n/quit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("r/Python", result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_repl_question_and_quit(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_engine.chat.return_value = MagicMock(
            answer="Paris is the capital.",
            sources=[{"subreddit": "france", "text": "Paris", "file_path": "/f.md"}],
        )
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(
            app, ["chat", "--no-stream"],
            input="What is the capital of France?\n/quit\n",
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("Paris is the capital.", result.output)

    @patch("rsi.cli._build_chat_engine")
    def test_repl_empty_input_ignored(self, mock_build):
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="\n\n/quit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)
        mock_engine.chat.assert_not_called()

    @patch("rsi.cli._build_chat_engine")
    def test_repl_exit_alias(self, mock_build):
        """Both /quit and /exit should terminate the REPL."""
        from rsi.cli import app

        mock_engine = MagicMock()
        mock_build.return_value = mock_engine

        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--no-stream"], input="/exit\n")
        self.assertEqual(result.exit_code, 0, msg=result.output)


class TestBuildChatEngine(unittest.TestCase):
    """Test the _build_chat_engine helper (integration of components)."""

    @patch("rsi.chat.providers.base.create_provider")
    @patch("rsi.indexer.search.SearchEngine")
    def test_build_creates_direct_engine(self, mock_search_cls, mock_create):
        from rsi.cli import _build_chat_engine

        mock_search_cls.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        engine = _build_chat_engine(
            provider="ollama",
            model="qwen2.5:7b",
            db_path=Path("/tmp/test.lance"),
            mode="hybrid",
            max_history=10,
            context_docs=5,
        )

        mock_search_cls.assert_called_once_with(db_path=Path("/tmp/test.lance"))
        mock_create.assert_called_once_with(provider="ollama", model="qwen2.5:7b")

        # Engine should be a DirectEngine
        from rsi.chat.engine import DirectEngine
        self.assertIsInstance(engine, DirectEngine)

    @patch("rsi.chat.providers.base.create_provider")
    @patch("rsi.indexer.search.SearchEngine")
    def test_build_passes_search_mode(self, mock_search_cls, mock_create):
        from rsi.cli import _build_chat_engine

        mock_search_cls.return_value = MagicMock()
        mock_create.return_value = MagicMock()

        engine = _build_chat_engine(
            provider="ollama",
            model="test",
            db_path=Path("/tmp/test.lance"),
            mode="semantic",
            max_history=5,
            context_docs=3,
        )

        from rsi.indexer.search import SearchMode
        self.assertEqual(engine._search_mode, SearchMode.SEMANTIC)


if __name__ == "__main__":
    unittest.main()
