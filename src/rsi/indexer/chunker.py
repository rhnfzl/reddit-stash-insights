"""Bounded passages with Reddit evidence identity and offsets into the parsed body."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

from rsi.core.identity import canonical_doc_id
from rsi.core.models import Comment, Post

_HEADER = re.compile(r'(?m)^(?P<quote>(?:> ?)*)\*\*Comment by (?P<author>[^\n]+)\*\*[^\n]*\n')
_LINK = re.compile(r'\[Permalink\]\(([^)]+)\)')
_PARAGRAPH = re.compile(r'\n[ \t]*\n')


def _fit(text: str, encode: Callable[[str], list[int]], budget: int) -> int:
    """Largest character prefix that fits, keeping UTF-8 sequences intact."""
    low, high = 0, min(len(text), max(1, budget))
    while len(encode(text[:high])) <= budget:
        if high == len(text):
            return high
        low, high = high, min(len(text), high * 2)
    while low < high:
        mid = (low + high + 1) // 2
        if len(encode(text[:mid])) <= budget:
            low = mid
        else:
            high = mid - 1
    return low


def _sections(item: Post | Comment) -> Iterator[tuple[int, int, str, str, str, str]]:
    kind = 'post' if isinstance(item, Post) else 'comment'
    evidence, parent = f'{kind}:{item.id}', getattr(item, 'parent_id', None) or ''
    start, url, context = 0, item.permalink, ''
    stack: dict[int, str] = {}
    for match in _HEADER.finditer(item.body):
        if match.start() > start:
            yield start, match.start(), evidence, parent, url, context
        depth = match['quote'].count('>')
        line_end = item.body.find('\n', match.end())
        link = _LINK.search(item.body[match.end():line_end if line_end >= 0 else len(item.body)])
        url = link[1] if link else ''
        parts = urlsplit(url).path.rstrip('/').split('/')
        evidence = f'comment:{parts[-1]}' if link and len(parts) >= 7 else f'{kind}:{item.id}'
        parent = stack.get(depth - 1, f'{kind}:{item.id}')
        stack = {level: value for level, value in stack.items() if level < depth}
        stack[depth] = evidence
        context = f'Comment by {match["author"]}\nParent: {parent}'
        start = match.start()
    if start < len(item.body) or not item.body:
        yield start, len(item.body), evidence, parent, url, context


def iter_chunks(
    item: Post | Comment, *, encode: Callable[[str], list[int]], decode: Callable[[list[int]], str],
    max_tokens: int = 512, overlap: int = 64,
) -> Iterator[dict[str, Any]]:
    """Yield bounded passages, preferring paragraph and comment boundaries.

    Offsets are character positions in ``item.body``. Character slicing avoids
    splitting Unicode characters when a tokenizer encodes one character as several tokens.
    """
    if max_tokens < 1 or overlap < 0 or overlap >= max_tokens:
        raise ValueError('chunk tokens must be positive and 0 <= overlap < chunk tokens')
    kind = 'post' if isinstance(item, Post) else 'comment'
    source = f'{kind}:{item.id}'
    thread = canonical_doc_id({'id': item.id, 'content_type': kind, 'permalink': item.permalink})
    title = item.title if isinstance(item, Post) else item.parent_title or ''
    generation = hashlib.sha256(f'{max_tokens}:{overlap}:{title}:{item.body}'.encode()).hexdigest()
    ordinal = 0
    for begin, end, evidence, parent, permalink, context in _sections(item):
        # Reserve at least half the passage for evidence even when a title is very long.
        heading = '\n'.join(part for part in (title, context) if part)
        heading = heading[:_fit(heading, encode, max_tokens // 2)]
        prefix = heading + '\n\n' if heading else ''
        budget = max_tokens - len(encode(prefix))
        if budget < 1:
            prefix, budget = '', max_tokens
        start = begin
        while start < end or begin == end:
            remaining = item.body[start:end]
            stop = start + _fit(remaining, encode, budget)
            if stop == start and remaining:
                raise ValueError('chunk token budget cannot fit one source character')
            if stop < end:
                boundary = None
                for boundary in _PARAGRAPH.finditer(item.body, start, stop):
                    pass
                if boundary is not None and boundary.end() > start:
                    stop = boundary.end()
            passage = item.body[start:stop]
            text = prefix + passage or " "
            # Token merges at the prefix boundary can change the exact count.
            while len(encode(text)) > max_tokens and stop > start:
                stop -= 1
                passage, text = item.body[start:stop], prefix + item.body[start:stop]
            if stop == start and remaining:
                raise ValueError('chunk token budget cannot fit evidence beside its context')
            yield {
                'id': item.id, 'doc_id': source, 'source_doc_id': source, 'evidence_id': evidence,
                'chunk_id': f'{source}:{ordinal}', 'ordinal': ordinal, 'generation': generation,
                'parent_doc_id': thread, 'parent_id': parent, 'text': text, 'permalink': permalink,
                'source_permalink': item.permalink, 'file_path': item.file_path, 'content_type': kind,
                'source_start': start, 'source_end': stop, 'source_offset_basis': 'body',
                'subreddit': item.subreddit, 'score': item.score,
                'timestamp': str(item.timestamp or ''), 'source': item.source,
            }
            ordinal += 1
            if stop >= end:
                break
            # Overlap only inside oversized paragraphs; whole paragraph boundaries remain independent.
            if item.body[max(start, stop - 2):stop] == '\n\n' or overlap == 0:
                start = stop
            else:
                tail_length = _fit(passage[::-1], lambda s: encode(s[::-1]), min(overlap, budget // 2))
                start = max(start + 1, stop - tail_length)
