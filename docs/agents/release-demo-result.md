# Synthetic wheel demo recording

The GIF shows a real local wheel installation, archive scan, keyword search, and chat answer with its source.
All three archive posts are synthetic. The keyword index is prepared before recording.
The chat uses the cached Qwen2.5 3B GGUF model through the relative path `models/qwen.gguf`.
No hosted provider or private archive is used. The local Qwen process has exited.

## Installation and import verification

The wheel is built from the workspace and installed in a fresh virtual environment outside the checkout.
Installation uses cached dependencies and index/chat extras. It does not claim a PyPI installation.
The build command is:

```bash
uv build --wheel --out-dir /private/tmp/rsi-demo-wheel-8Xt2CQ/dist --cache-dir /private/tmp/rsi-uv-cache --offline
```

The installed import resolves to:

```text
/private/tmp/rsi-demo-wheel-8Xt2CQ/.venv/lib/python3.14/site-packages/rsi/__init__.py
```

## Captured commands and standard output

Commands run with the fresh virtual environment active, from `/private/tmp/rsi-demo-wheel-8Xt2CQ`.
`RSI_DB_PATH=index` selects the copied synthetic index. Cloud fallback is disabled.
The following text comes from the recorded subprocess output. Every command exits zero.

```console
$ uv pip install --offline --quiet './reddit_stash_insights-0.1.0-py3-none-any.whl[index,chat]'
```

```console
$ python -c 'from importlib.metadata import version; print(version("reddit-stash-insights"))'
0.1.0
```

```console
$ rsi scan corpus
Found 3 posts and 0 comments
```

```console
$ rsi search 'Cedar backup rota' --mode keyword --limit 1
1 results for: Cedar backup rota

  1. [r/homelab] (score: 0) Cedar backup rota  # Cedar backup rota  The Cedar backup job starts at 03:40 UTC.
     File: corpus/homelab/POST_f001.md
```

```console
$ rsi chat 'When does the Cedar backup job start?' --mode keyword --provider llama-cpp --model models/qwen.gguf --limit 1 --no-stream
Chat context destination: llama-cpp (in process) (model models/qwen.gguf)
The Cedar backup job starts at 03:40 UTC. [1]

  [1] r/homelab -- corpus/homelab/POST_f001.md | https://reddit.com/r/homelab/comments/f001/synthetic/
      Cedar backup rota  # Cedar backup rota  The Cedar backup job starts at 03:40 UTC.
```

## Artifact checks

The GIF has four frames at 1080 by 600 pixels. Durations are 6.5, 4, 6.5, and 10 seconds, totaling 27 seconds.
The file is 83,621 bytes. Source and answer text are visible in the decoded final frame.
Layout assertions reject text clipping. The displayed output contains no private home paths.
The complete command transcript is `/private/tmp/rsi-demo-wheel-8Xt2CQ/transcript.json`.
The rendering script and validation JSON remain in the same temporary directory.
Only the GIF and this working note change in the repository.

GIF SHA-256: `a1c03cc4d8196d05bbb70b8dea964ba272886bf6b03378e01e048dbb2595a910`.
Demo wheel SHA-256: `9a15475d7058494079427b463e8620aa5fd1e9d19793e9272fa0693007cfcc6c`.

The demo uses prepared data and cached dependencies. A clean online installation is not tested here.
Final release archives must be rebuilt after this GIF change.
