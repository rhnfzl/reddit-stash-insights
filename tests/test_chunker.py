"""Passage boundaries and original source identity without model downloads."""
import unittest

from rsi.core.models import Comment, Post


def encode(text):
    return list(text.encode('utf-8'))


def decode(tokens):
    return bytes(tokens).decode('utf-8')


class TestChunker(unittest.TestCase):
    def chunks(self, item, **kwargs):
        from rsi.indexer.chunker import iter_chunks
        return list(iter_chunks(item, encode=encode, decode=decode, **kwargs))

    def post(self, body):
        return Post('p', 'test', 'Thread title', body, 'https://reddit.com/r/test/comments/p/title/', 'POST_p.md')

    def test_late_unicode_and_code_evidence_is_retained_with_exact_offsets(self):
        body = 'early paragraph\n\n' + ('```\n' + '日本語🙂 = value\n' * 3000 + '```\n\nlate needle')
        rows = self.chunks(self.post(body))
        self.assertGreater(len(rows), 10)
        self.assertIn('late needle', rows[-1]['text'])
        for row in rows:
            self.assertLessEqual(len(encode(row['text'])), 512)
            self.assertIn(body[row['source_start']:row['source_end']], row['text'])
            self.assertEqual(row['id'], 'p')
            self.assertEqual(row['parent_doc_id'], 'post:p')
        self.assertEqual(rows, self.chunks(self.post(body)))
        self.assertEqual(len({r['chunk_id'] for r in rows}), len(rows))

    def test_nested_comments_preserve_evidence_parent_url_and_boundary(self):
        body = ('Post body\n\n**Comment by /u/a**\n'
                '*Upvotes: 1 | [Permalink](https://reddit.com/r/test/comments/p/title/c1/)*\n\nfirst evidence\n'
                '> **Comment by /u/b**\n'
                '> *Upvotes: 2 | [Permalink](https://reddit.com/r/test/comments/p/title/c2/)*\n\n'
                '> child evidence\n')
        rows = self.chunks(self.post(body))
        child = next(r for r in rows if 'child evidence' in r['text'])
        self.assertEqual(child['evidence_id'], 'comment:c2')
        self.assertEqual(child['parent_id'], 'comment:c1')
        self.assertTrue(child['permalink'].endswith('/c2/'))
        self.assertEqual(child['parent_doc_id'], 'post:p')
        self.assertNotIn('first evidence', child['text'])

    def test_standalone_comment_collapses_to_permalink_thread(self):
        item = Comment('c1', 'test', 'a', 'standalone evidence',
                       'https://reddit.com/r/test/comments/p/title/c1/', 'COMMENT_c1.md', parent_title='Thread')
        row = self.chunks(item)[0]
        self.assertEqual((row['id'], row['source_doc_id'], row['evidence_id'], row['parent_doc_id']),
                         ('c1', 'comment:c1', 'comment:c1', 'post:p'))

    def test_missing_permalink_is_not_invented(self):
        row = self.chunks(Comment('c', 'test', 'a', 'body', '', 'COMMENT_c.md'))[0]
        self.assertEqual(row['permalink'], '')
        self.assertEqual(row['parent_doc_id'], 'comment:c')

    def test_invalid_settings_raise(self):
        for tokens, overlap in [(0, 0), (10, -1), (10, 10)]:
            with self.assertRaises(ValueError):
                self.chunks(self.post('body'), max_tokens=tokens, overlap=overlap)


class TestChunkSettings(unittest.TestCase):
    def test_config_and_env_chunk_settings(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from rsi.config import Settings
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.toml"
            config.write_text("[index]\nchunk_tokens=256\nchunk_overlap=32\n")
            with patch.dict("os.environ", {}, clear=True):
                settings = Settings.load(config)
                self.assertEqual((settings.chunk_tokens, settings.chunk_overlap), (256, 32))
                with patch.dict("os.environ", {"RSI_CHUNK_TOKENS": "128", "RSI_CHUNK_OVERLAP": "16"}):
                    settings = Settings.load(config)
                    self.assertEqual((settings.chunk_tokens, settings.chunk_overlap), (128, 16))
                for env in ({"RSI_CHUNK_TOKENS": "0"}, {"RSI_CHUNK_OVERLAP": "256"}):
                    with patch.dict("os.environ", env), self.assertRaises(ValueError):
                        Settings.load(config)

class TestEmptyBody(unittest.TestCase):
    def test_title_only_post_retains_one_source_passage(self):
        from rsi.indexer.chunker import iter_chunks
        item = Post('p', 'test', 'Only title', '', '', 'POST_p.md')
        rows = list(iter_chunks(item, encode=encode, decode=decode))
        self.assertEqual(len(rows), 1)
        self.assertIn('Only title', rows[0]['text'])
        self.assertEqual((rows[0]['source_start'], rows[0]['source_end']), (0, 0))
