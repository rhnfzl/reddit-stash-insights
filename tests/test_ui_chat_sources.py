"""Source evidence stays visible in the installed chat interface."""
import unittest
from pathlib import Path

try:
    import lancedb  # noqa: F401
    from streamlit.testing.v1 import AppTest
    HAS_UI = True
except ImportError:
    HAS_UI = False


@unittest.skipUnless(HAS_UI, "requires ui extra")
class TestUISources(unittest.TestCase):
    def test_saved_answer_shows_evidence_identity_and_offsets(self):
        from rsi.config import Settings

        app = AppTest.from_file(str(Path(__file__).parents[1] / "src/rsi/ui/chat.py"))
        app.session_state.settings = Settings(db_path=Path("/nonexistent/synthetic-index"))
        app.session_state.messages = [{"role": "assistant", "content": "Answer [1]", "sources": [{
            "text": "evidence", "file_path": "/archive/thread.md", "evidence_id": "comment:answer",
            "chunk_id": "post:thread:4", "source_start": 600, "source_end": 612,
            "permalink": "https://reddit.com/evidence",
            "evidence_refs": [
                {"evidence_id": "comment:a", "permalink": "https://reddit.com/a", "parent_id": "post:thread",
                 "source_start": 600, "source_end": 605},
                {"evidence_id": "comment:b", "permalink": "https://reddit.com/b", "parent_id": "comment:a",
                 "source_start": 605, "source_end": 612},
            ],
        }]}]
        app.run()
        self.assertFalse(app.exception)
        rendered = "\n".join(markdown.value for markdown in app.markdown)
        for value in ("[1]", "/archive/thread.md", "comment:answer", "post:thread:4", "600:612",
                      "https://reddit.com/evidence", "comment:a", "comment:b", "post:thread",
                      "https://reddit.com/a", "https://reddit.com/b", "600:605", "605:612"):
            self.assertIn(value, rendered)
