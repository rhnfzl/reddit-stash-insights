"""Tests for CLI commands."""
import unittest
import tempfile
from typer.testing import CliRunner


class TestCLI(unittest.TestCase):
    def test_version_flag(self):
        from rsi.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("0.1.0", result.output)

    def test_scan_command_on_empty_dir(self):
        from rsi.cli import app

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(app, ["scan", tmpdir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("0 posts", result.output)
            self.assertIn("0 comments", result.output)

    def test_scan_command_on_nonexistent_dir(self):
        from rsi.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["scan", "/nonexistent/path"])
        self.assertNotEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
