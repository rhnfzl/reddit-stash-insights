"""Passage boundaries and original source identity without model downloads."""
import unittest

from rsi.core.models import Comment, Post


def encode(text):
    return list(text.encode('utf-8'))


def decode(tokens):
    return bytes(tokens).decode('utf-8')


class ChunkerTestCase(unittest.TestCase):
    def chunks(self, item, **kwargs):
        from rsi.indexer.chunker import iter_chunks
        return list(iter_chunks(item, encode=encode, decode=decode, **kwargs))

    def post(self, body):
        return Post('p', 'test', 'Thread title', body, 'https://reddit.com/r/test/comments/p/title/', 'POST_p.md')


class TestChunker(ChunkerTestCase):
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
        ref = next(ref for ref in child['evidence_refs'] if ref['evidence_id'] == 'comment:c2')
        self.assertEqual(ref['parent_id'], 'comment:c1')
        self.assertTrue(ref['permalink'].endswith('/c2/'))
        self.assertEqual(child['parent_doc_id'], 'post:p')
        self.assertIn('child evidence', body[ref['source_start']:ref['source_end']])

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

class TestPackedEvidence(ChunkerTestCase):
    def body(self):
        return ('Post body\n\n**Comment by /u/a**\n'
                '*Upvotes: 1 | [Permalink](https://reddit.com/r/test/comments/p/title/c1/)*\n\nfirst evidence\n'
                '> **Comment by /u/b**\n'
                '> *Upvotes: 2 | [Permalink](https://reddit.com/r/test/comments/p/title/c2/)*\n\n'
                '> child evidence\n'
                '**Comment by /u/c**\n'
                '*Upvotes: 3 | [Permalink](https://reddit.com/r/test/comments/p/title/c3/)*\n\nlast sibling\n')

    def test_short_nested_and_sibling_comments_share_bounded_passage_with_all_refs(self):
        body = self.body()
        rows = self.chunks(self.post(body), max_tokens=1024)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['evidence_id'], 'post:p')
        self.assertEqual(row['permalink'], self.post(body).permalink)
        refs = row['evidence_refs']
        self.assertEqual([r['evidence_id'] for r in refs], ['post:p', 'comment:c1', 'comment:c2', 'comment:c3'])
        self.assertEqual([r['parent_id'] for r in refs], ['', 'post:p', 'comment:c1', 'post:p'])
        self.assertEqual(''.join(body[r['source_start']:r['source_end']] for r in refs), body)

    def test_all_rows_have_nonempty_clipped_evidence_and_full_source_coverage(self):
        body = self.body() + ('```\n日本語🙂code\n' * 300) + '\nlate needle'
        rows = self.chunks(self.post(body))
        covered: set[int] = set()
        for row in rows:
            self.assertLessEqual(len(encode(row['text'])), 512)
            self.assertTrue(row['evidence_refs'])
            self.assertIn(body[row['source_start']:row['source_end']], row['text'])
            for ref in row['evidence_refs']:
                self.assertGreaterEqual(ref['source_start'], row['source_start'])
                self.assertLessEqual(ref['source_end'], row['source_end'])
                covered.update(range(ref['source_start'], ref['source_end']))
        self.assertEqual(covered, set(range(len(body))))
        self.assertIn('late needle', rows[-1]['text'])

    def test_single_comment_scalar_fields_stay_exact_and_refs_are_nonempty(self):
        item = Comment('c1', 'test', 'a', 'body', 'https://reddit.com/r/test/comments/p/title/c1/', 'COMMENT_c1.md')
        row = self.chunks(item)[0]
        self.assertEqual(row['evidence_id'], 'comment:c1')
        self.assertEqual(row['permalink'], item.permalink)
        self.assertEqual(row['evidence_refs'], [{'evidence_id': 'comment:c1', 'permalink': item.permalink,
                                               'parent_id': '', 'source_start': 0, 'source_end': 4}])

    def test_packed_comment_archive_uses_thread_permalink(self):
        item = Comment('c1', 'test', 'a', self.body(),
                       'https://reddit.com/r/test/comments/p/title/c1/', 'COMMENT_c1.md')
        rows = self.chunks(item, max_tokens=1024)
        self.assertEqual(rows[0]['evidence_id'], 'post:p')
        self.assertEqual(rows[0]['permalink'], 'https://reddit.com/r/test/comments/p/title/')

    def test_packed_thread_permalink_without_trailing_slash_is_retained(self):
        body = self.body().replace('https://reddit.com/r/test/comments/p/title/', 'https://example.com/')
        item = Post('p', 'test', 'Title', body, 'https://reddit.com/r/test/comments/p/title', 'POST_p.md')
        row = self.chunks(item, max_tokens=1024)[0]
        self.assertEqual(row['permalink'].rstrip('/'), item.permalink)

class TestBoundedSectionProbe(unittest.TestCase):
    def test_oversized_sections_are_probed_without_full_tokenization(self):
        from rsi.indexer.chunker import iter_chunks
        body = ('small introduction\n\n**Comment by /u/a**\n'
                '*Upvotes: 1 | [Permalink](https://reddit.com/r/test/comments/p/title/c1/)*\n\n'
                + 'x' * 100_000)
        item = Post('p', 'test', 'Title', body, 'https://reddit.com/r/test/comments/p/title/', 'POST_p.md')
        sizes = []
        def spy(text):
            sizes.append(len(text))
            return encode(text)
        covered = 0
        rows = 0
        for row in iter_chunks(item, encode=spy, decode=decode):
            self.assertLessEqual(len(encode(row['text'])), 512)
            self.assertTrue(row['evidence_refs'])
            self.assertLessEqual(row['source_start'], covered)
            covered = max(covered, row['source_end'])
            rows += 1
        self.assertGreater(rows, 100)
        self.assertEqual(covered, len(body))
        self.assertLessEqual(max(sizes), 2 * 512 + len('Title\n\n'))

class TestAdvancingPassages(unittest.TestCase):
    def assert_ordered_coverage(self, rows, body, tokenizer):
        end = 0
        for row in rows:
            self.assertLessEqual(row['source_start'], end)
            self.assertGreater(row['source_end'], end)
            self.assertLessEqual(len(tokenizer(row['text'])), 512)
            self.assertTrue(row['evidence_refs'])
            end = row['source_end']
        self.assertEqual(end, len(body))

    def test_whitespace_paragraph_separator_does_not_repeat_overlapped_boundary(self):
        from rsi.indexer.chunker import iter_chunks
        body = 'a' * 300 + '\n \n' + 'b' * 1200
        item = Post('p', 'test', 'Title', body, '', 'POST_p.md')
        rows = list(iter_chunks(item, encode=encode, decode=decode))
        self.assert_ordered_coverage(rows, body, encode)

    def test_nonmonotonic_section_fit_flushes_pending_before_oversized_section(self):
        from rsi.indexer.chunker import iter_chunks
        intro = 'INTRO\n\n'
        section = ('**Comment by /u/a**\n'
                   '*Upvotes: 1 | [Permalink](https://reddit.com/r/test/comments/p/title/c1/)*\n\n' + 'x' * 900)
        body = intro + section
        def merging_encoder(text):
            return [0] * (len(text) // 8 + 1) if text.startswith(intro) else encode(text)
        self.assertLessEqual(len(merging_encoder(body)), 512)
        self.assertGreater(len(merging_encoder(section)), 512)
        item = Post('p', 'test', '', body, '', 'POST_p.md')
        rows = list(iter_chunks(item, encode=merging_encoder, decode=decode))
        self.assert_ordered_coverage(rows, body, merging_encoder)
        self.assertEqual(rows[0]['source_end'], len(intro))
