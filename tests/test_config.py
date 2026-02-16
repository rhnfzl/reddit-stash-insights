"""Tests for configuration system."""
import unittest
import tempfile
from pathlib import Path


class TestSettings(unittest.TestCase):
    def test_default_settings(self):
        from rsi.config import Settings

        s = Settings()
        self.assertEqual(s.embedding_model, "BAAI/bge-m3")
        self.assertEqual(s.llm_provider, "llama-cpp")
        self.assertEqual(s.llm_model, "qwen2.5:7b")
        self.assertIsNone(s.s3_bucket)

    def test_load_from_toml(self):
        from rsi.config import Settings

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            f.write(b'[embedding]\nmodel = "custom/model"\n\n[llm]\nprovider = "openai"\nmodel = "gpt-4o-mini"\n')
            config_path = Path(f.name)

        try:
            s = Settings.load(config_path=config_path)
            self.assertEqual(s.embedding_model, "custom/model")
            self.assertEqual(s.llm_provider, "openai")
            self.assertEqual(s.llm_model, "gpt-4o-mini")
        finally:
            config_path.unlink()

    def test_env_var_overrides_toml(self):
        import os
        from rsi.config import Settings

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            f.write(b'[embedding]\nmodel = "toml/model"\n')
            config_path = Path(f.name)

        try:
            os.environ["RSI_EMBEDDING_MODEL"] = "env/override"
            s = Settings.load(config_path=config_path)
            self.assertEqual(s.embedding_model, "env/override")
        finally:
            config_path.unlink()
            del os.environ["RSI_EMBEDDING_MODEL"]

    def test_load_without_config_file(self):
        from rsi.config import Settings

        s = Settings.load(config_path=Path("/nonexistent/config.toml"))
        self.assertEqual(s.embedding_model, "BAAI/bge-m3")  # Falls back to defaults


if __name__ == "__main__":
    unittest.main()
