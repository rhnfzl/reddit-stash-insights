"""Installed UI launching does not depend on the current directory."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from rsi.cli import app


class TestUILaunch(unittest.TestCase):
    def test_missing_ui_dependency_is_actionable(self):
        with patch("importlib.util.find_spec", return_value=None):
            result = CliRunner().invoke(app, ["ui"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("[ui]", result.output)

    def test_launch_uses_installed_script_and_propagates_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                with patch("importlib.util.find_spec", return_value=object()), patch("subprocess.run") as run:
                    run.return_value.returncode = 7
                    result = CliRunner().invoke(app, ["ui", "--port", "8509"])
            finally:
                os.chdir(previous)
        self.assertEqual(result.exit_code, 7)
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertTrue(Path(command[4]).is_file())
        self.assertEqual(command[5:], ["--server.address", "127.0.0.1", "--server.port", "8509"])

    def test_invalid_port_fails_before_launch(self):
        with patch("subprocess.run") as run:
            result = CliRunner().invoke(app, ["ui", "--port", "0"])
        self.assertEqual(result.exit_code, 2)
        self.assertIn("range", result.output)
        run.assert_not_called()
