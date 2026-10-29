# Remote MCP server and indexer using an OpenAI-compatible embeddings API.
# No torch and no local model: embeddings come from RSI_EMBEDDING_BASE_URL / OPENAI_BASE_URL.
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/rsi \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    RSI_EMBEDDING_PROVIDER=openai \
    RSI_LLM_PROVIDER=openai \
    RSI_LLM_MODEL=gpt-4o-mini \
    RSI_DB_PATH=/data/index

RUN useradd --uid 1000 --create-home --home-dir /home/rsi rsi

WORKDIR /app

# Dependency layer: only invalidated by pyproject.toml (the weekly scheduled build skips the cache).
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/rsi && echo '__version__ = "0.0.0"' > src/rsi/__init__.py \
    && pip install ".[index,embed-openai,mcp,chat-openai]" \
    && pip uninstall -y reddit-stash-insights \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" \
    && chmod -R a+rX /opt/tiktoken

COPY src ./src
RUN pip install --no-deps . && rm -rf /app/src \
    && mkdir -p /data && chown 1000:1000 /data

USER 1000
VOLUME ["/data"]
EXPOSE 8765

HEALTHCHECK --interval=60s --timeout=20s --start-period=30s --retries=3 CMD ["rsi", "health"]

ENTRYPOINT ["rsi"]
CMD ["mcp", "--transport", "http", "--host", "0.0.0.0", "--port", "8765"]
