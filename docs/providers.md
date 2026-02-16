# LLM Provider Setup

reddit-stash-insights supports three LLM providers for RAG chat. Each provider is an optional dependency — install only what you need.

## Provider Comparison

| | llama-cpp | Ollama | OpenAI-compatible |
|---|---|---|---|
| **Type** | Local (in-process) | Local (HTTP server) | Cloud / API |
| **Privacy** | Full (no network) | Full (localhost) | Data sent to API |
| **Setup** | Download GGUF file | Install Ollama + pull model | Set API key |
| **Cost** | Free | Free | Per-token pricing |
| **Latency** | Low (GPU) / Medium (CPU) | Low-Medium | Depends on network |
| **Multi-model** | One model at a time | Multiple models ready | Any model via API |
| **GPU** | Metal (macOS), CUDA (Linux) | Automatic | N/A |
| **Install extra** | `chat` | `chat-ollama` | `chat-openai` |

## llama-cpp (Local GGUF)

Best for privacy-first, offline usage. Runs the model directly in-process via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).

### Setup

```bash
# 1. Install the provider
pip install reddit-stash-insights[search,chat]

# 2. Download a GGUF model
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf \
  --local-dir ~/.rsi/models/

# 3. Configure (optional — auto-detects GGUF files in ~/.rsi/models/)
cat > ~/.rsi/config.toml << 'EOF'
[llm]
provider = "llama-cpp"
model = "~/.rsi/models/qwen2.5-7b-instruct-q4_k_m.gguf"
EOF
```

### GGUF Auto-Detection

If the `model` setting isn't a valid file path, rsi scans `~/.rsi/models/` for the first `.gguf` file (sorted alphabetically). This means you can just drop a GGUF into the directory and it works.

Split GGUF files (e.g., `model-00001-of-00003.gguf`) are also detected.

### GPU Acceleration

- **macOS (Apple Silicon):** Metal acceleration is automatic with `llama-cpp-python`. No extra configuration needed.
- **Linux (NVIDIA):** Install with CUDA support: `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python`

### Model Recommendations

| Model | Size | RAM Required | Quality |
|-------|------|-------------|---------|
| Qwen2.5-7B-Instruct Q4_K_M | ~4.4 GB | ~6-7 GB | Good for RAG |
| Qwen2.5-3B-Instruct Q4_K_M | ~2 GB | ~3-4 GB | Basic, may miss nuance |
| Qwen2.5-14B-Instruct Q4_K_M | ~8 GB | ~10-12 GB | High quality, needs 16GB+ RAM |

### Parameters

The llama-cpp provider accepts these constructor parameters (via config or code):

- `n_ctx=4096` — context window size
- `n_gpu_layers=-1` — layers to offload to GPU (-1 = all)
- `temperature=0.3` — generation temperature
- `max_tokens=512` — maximum response tokens

## Ollama (Local HTTP)

Best for easy model management with multiple models. Runs as a background service with an HTTP API.

### Setup

```bash
# 1. Install Ollama (https://ollama.ai)
# macOS:
brew install ollama

# 2. Pull a model
ollama pull qwen2.5:7b

# 3. Start the server (runs on localhost:11434)
ollama serve

# 4. Install the provider
pip install reddit-stash-insights[search,chat-ollama]

# 5. Configure
cat > ~/.rsi/config.toml << 'EOF'
[llm]
provider = "ollama"
model = "qwen2.5:7b"
EOF
```

### Verification

Check that Ollama is running and the model is available:

```bash
curl http://localhost:11434/api/tags
```

The availability checker pings this endpoint before attempting to use Ollama. If the server isn't running, rsi falls back to the next available provider.

## OpenAI-Compatible (Cloud / API)

Works with OpenAI, vLLM, LM Studio, and any server that implements the OpenAI chat completions API.

### Setup with OpenAI

```bash
# 1. Install the provider
pip install reddit-stash-insights[search,chat-openai]

# 2. Set your API key
export OPENAI_API_KEY=sk-...

# 3. Configure
cat > ~/.rsi/config.toml << 'EOF'
[llm]
provider = "openai"
model = "gpt-4o"
EOF
```

### Setup with vLLM / LM Studio

For self-hosted OpenAI-compatible servers, set the `OPENAI_BASE_URL` environment variable:

```bash
# vLLM
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy  # vLLM doesn't validate keys

# LM Studio
export OPENAI_BASE_URL=http://localhost:1234/v1
export OPENAI_API_KEY=dummy
```

Then configure as normal:

```toml
[llm]
provider = "openai"
model = "Qwen/Qwen2.5-7B-Instruct"  # Model name as loaded in vLLM
```

## Auto-Fallback

When starting a chat session, rsi pre-flights the configured provider to verify it's available. If the selected provider fails, it tries alternatives in this order:

1. **llama-cpp** — checks that `llama-cpp-python` is importable and a GGUF file exists
2. **ollama** — pings `http://localhost:11434/api/tags`
3. **openai** — checks that the `openai` package is importable and `OPENAI_API_KEY` is set

The fallback is transparent — rsi logs which provider it switched to and why. In the Streamlit UI, a note appears when a fallback provider is used.

### Example Fallback Scenarios

| Configured | What happens | Falls back to |
|-----------|-------------|---------------|
| `llama-cpp` | No GGUF file in `~/.rsi/models/` | `ollama` (if running) |
| `ollama` | Ollama server not running | `llama-cpp` (if GGUF available) |
| `openai` | No API key set | `llama-cpp` (if GGUF available) |
| `llama-cpp` | Package not installed | `ollama` -> `openai` |

### Disabling Fallback

To use a specific provider without fallback, pass `--provider` on the CLI:

```bash
rsi chat --provider ollama "my question"
```

The `--provider` flag overrides config but still participates in fallback if unavailable. To force a specific provider (no fallback), ensure only one provider's dependencies are installed.
