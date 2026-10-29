"""The exact container deployment commands, run against a fixture archive (fake embedder, no network).

Keep DEPLOY in sync with the documented deployment contract: the CLI is the source of truth.
"""
import os
import shlex
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from tests.test_index_build import HAS_LANCEDB, FakeEmbedder, make_reddit_dir

# (command as an operator types it after the `rsi` entrypoint, expected exit code)
DEPLOY = [
    ("index /src --db-path /data/next --rebuild --max-error-pct 1", 0),  # build into staging
    ("health --db-path /data/next --mode hybrid", 0),  # staging check, exercises embeddings
    ("health --db-path /data/next", 0),  # container healthcheck form
    ("health", 0),  # image HEALTHCHECK, uses RSI_DB_PATH
    # Interfaces that do NOT exist, so nobody deploys them: the archive dir is positional, there is no `check`.
    ("index --db-path /data/next --rebuild", 2),
    ("check --db /data/next", 2),
]


@unittest.skipUnless(HAS_LANCEDB, "requires index extra (lancedb)")
class TestDeployContract(unittest.TestCase):
    def test_exact_deployment_commands(self):
        from rsi.cli import app

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = make_reddit_dir(root)
            data = root / "data"
            data.mkdir()

            def local(arg):
                # Map the container mount points onto the fixture; everything else is passed verbatim.
                for container, host in (("/src", src), ("/data", data)):
                    if arg == container or arg.startswith(container + "/"):
                        return str(host) + arg[len(container):]
                return arg

            env = {"RSI_EMBEDDING_PROVIDER": "openai", "RSI_DB_PATH": str(data / "next")}
            for command, expected in DEPLOY:
                args = [local(a) for a in shlex.split(command)]
                with patch("rsi.indexer.search.make_embedder", return_value=FakeEmbedder()), \
                        patch.dict(os.environ, env):
                    result = CliRunner().invoke(app, args)
                self.assertEqual(result.exit_code, expected, msg=f"rsi {command}\n{result.output}")


if __name__ == "__main__":
    unittest.main()
