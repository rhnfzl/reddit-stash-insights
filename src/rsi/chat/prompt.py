"""RAG prompt templates for building LLM messages."""
from __future__ import annotations

from rsi.chat.providers.base import ChatMessage

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about the user's saved Reddit content. "
    "Use ONLY the provided context to answer. If the context doesn't contain relevant information, say so. "
    "Always cite sources using their reference number [1], [2], etc. Be concise."
)


def build_context_block(docs: list[dict], max_text_len: int = 500) -> str:
    """Format retrieved docs as a numbered context block for the LLM."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        sub = doc.get("subreddit", "?")
        score = doc.get("score", 0)
        ctype = doc.get("content_type", "?")
        text = doc.get("text", "")[:max_text_len]
        blocks.append(f"[{i}] r/{sub} | {ctype} | Score: {score}\n{text}")
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    docs: list[dict],
    history: list[ChatMessage],
    system: str = SYSTEM_PROMPT,
) -> list[ChatMessage]:
    """Assemble full message list: system + history + context + query."""
    messages = [ChatMessage(role="system", content=system)]
    messages.extend(history)
    context = build_context_block(docs)
    messages.append(ChatMessage(role="user", content=f"CONTEXT:\n{context}\n\nQUESTION: {query}"))
    return messages
