"""Bounded passages with Reddit evidence identity and offsets into the parsed body."""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from rsi.core.identity import canonical_doc_id
from rsi.core.models import Comment, Post

ALGORITHM_VERSION = 3
Section = tuple[int, int, str, str, str, str]

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


def _sections(item: Post | Comment) -> Iterator[Section]:
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


def _thread_permalink(permalink: str) -> str:
    parts = urlsplit(permalink)
    if parts.scheme not in ("http", "https") or not (parts.hostname == "reddit.com"
                                                     or (parts.hostname or "").endswith(".reddit.com")):
        return ""
    match = re.match(r"(.*/comments/[a-zA-Z0-9_-]+(?:/[^/]+)?)(?:/|$)", parts.path)
    return urlunsplit((parts.scheme, parts.netloc, match[1].rstrip("/") + "/", "", "")) if match else ""


def _packed_sections(
    item: Post | Comment, encode: Callable[[str], list[int]], prefix: str, max_tokens: int,
) -> Iterator[list[Section]]:
    """Keep one bounded buffer of adjacent complete evidence sections."""
    pending: list[Section] = []
    for section in _sections(item):
        text = item.body[section[0]:section[1]]
        if _fit(text, lambda body: encode(prefix + body), max_tokens) < len(text):
            if pending:
                yield pending
                pending = []
            yield [section]
            continue
        if pending:
            candidate = item.body[pending[0][0]:section[1]]
            if _fit(candidate, lambda text: encode(prefix + text), max_tokens) < len(candidate):
                yield pending
                pending = []
        pending.append(section)
    if pending:
        yield pending


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
    generation = hashlib.sha256(f'{ALGORITHM_VERSION}:{max_tokens}:{overlap}:{title}:{item.body}'.encode()).hexdigest()
    ordinal = 0
    packed_title = title[:_fit(title, encode, max_tokens // 2)]
    packed_prefix = packed_title + '\n\n' if packed_title else ''
    for sections in _packed_sections(item, encode, packed_prefix, max_tokens):
        begin, end = sections[0][0], sections[-1][1]
        _, _, evidence, parent, permalink, context = sections[0]
        if len(sections) > 1:
            evidence, parent, context = thread, '', ''
            permalink = _thread_permalink(item.permalink)
            if not permalink:
                permalink = next((_thread_permalink(section[4]) for section in sections
                                  if _thread_permalink(section[4])), '')
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
            paragraph_end = None
            if stop < end:
                boundary = None
                for boundary in _PARAGRAPH.finditer(item.body, start, stop):
                    pass
                if boundary is not None and boundary.end() > start:
                    paragraph_end = boundary.end()
                    stop = paragraph_end
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
                'evidence_refs': [
                    {'evidence_id': section[2], 'permalink': section[4], 'parent_id': section[3],
                     'source_start': max(start, section[0]), 'source_end': min(stop, section[1])}
                    for section in sections if section[0] < stop and section[1] > start or begin == end
                ],
                'source_permalink': item.permalink, 'file_path': item.file_path, 'content_type': kind,
                'source_start': start, 'source_end': stop, 'source_offset_basis': 'body',
                'subreddit': item.subreddit, 'score': item.score,
                'timestamp': str(item.timestamp or ''), 'source': item.source,
            }
            ordinal += 1
            if stop >= end:
                break
            # Overlap only inside oversized paragraphs; whole paragraph boundaries remain independent.
            if stop == paragraph_end or overlap == 0:
                start = stop
            else:
                tail_length = _fit(passage[::-1], lambda s: encode(s[::-1]), min(overlap, budget // 2))
                start = max(start + 1, stop - tail_length)
