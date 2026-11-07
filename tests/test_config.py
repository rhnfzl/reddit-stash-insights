"""Tests for configuration system."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import os


class TestSettings(unittest.TestCase):
    def test_cloud_fallback_defaults_off(self):
        from rsi.config import Settings

        self.assertFalse(Settings().allow_cloud_fallback)

    def test_cloud_fallback_config_and_env_override(self):
        from rsi.config import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.toml"
            path.write_text("[llm]\nallow_cloud_fallback = true\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertTrue(Settings.load(path).allow_cloud_fallback)
            with patch.dict(os.environ, {"RSI_ALLOW_CLOUD_FALLBACK": "false"}):
                self.assertFalse(Settings.load(path).allow_cloud_fallback)

    def test_cloud_fallback_env_requires_boolean(self):
        from rsi.config import Settings

        for value in ("true", "false"):
            with self.subTest(value=value), patch.dict(os.environ, {"RSI_ALLOW_CLOUD_FALLBACK": value}):
                self.assertEqual(Settings.load(Path("/nonexistent/config.toml")).allow_cloud_fallback, value == "true")
        for value in ("yes", "1", "", "invalid"):
            with self.subTest(value=value), patch.dict(os.environ, {"RSI_ALLOW_CLOUD_FALLBACK": value}):
                with self.assertRaisesRegex(ValueError, "RSI_ALLOW_CLOUD_FALLBACK"):
                    Settings.load(Path("/nonexistent/config.toml"))

    def test_cloud_fallback_config_requires_boolean(self):
        from rsi.config import Settings

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.toml"
            for value in ('"false"', "1"):
                path.write_text(f"[llm]\nallow_cloud_fallback = {value}\n")
                with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(ValueError, "allow_cloud_fallback"):
                        Settings.load(path)

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

    def test_non_positive_index_chunk_rejected(self):
        import os
        from unittest.mock import patch

        from rsi.config import Settings

        for bad in ("0", "-1"):
            with patch.dict(os.environ, {"RSI_INDEX_CHUNK": bad}), self.assertRaises(ValueError):
                Settings.load(config_path=Path("/nonexistent/config.toml"))

    def test_image_env_resolves_a_valid_openai_chat_model(self):
        """The Dockerfile's own ENV, with no extra variables, must not pair openai with a local model name."""
        import os
        import re
        from unittest.mock import patch

        from rsi.config import Settings

        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        image_env = dict(re.findall(r"\b(RSI_\w+)=(\S+)", dockerfile.read_text()))
        with patch.dict(os.environ):
            for k in [k for k in os.environ if k.startswith("RSI_")]:
                del os.environ[k]
            os.environ.update(image_env)
            os.environ["RSI_DB_PATH"] = tempfile.mkdtemp()  # /data/index does not exist outside the image
            s = Settings.load(config_path=Path("/nonexistent/config.toml"))
        self.assertEqual((s.llm_provider, s.llm_model), ("openai", "gpt-4o-mini"))
        self.assertEqual((s.embedding_provider, s.embedding_model), ("openai", "text-embedding-3-small"))


if __name__ == "__main__":
    unittest.main()
