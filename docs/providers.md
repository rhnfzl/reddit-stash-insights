# LLM Provider Setup

You can chat with your archive using local models or an OpenAI-compatible API.
Local provider failure does not permit cloud fallback unless you opt in.
Each provider has an optional dependency. Install only what you need.
Use the [source checkout instructions](../README.md#source-installation), then run installation commands from that checkout.

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
pip install -e ".[search,chat]"

# 2. Download a GGUF model
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m.gguf \
  --local-dir ~/.rsi/models/

# 3. Configure (optional, auto-detects GGUF files in ~/.rsi/models/)
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

The llama-cpp provider accepts these constructor parameters in Python code:

- `n_ctx=4096`: context window size
- `n_gpu_layers=-1`: layers to offload to GPU (-1 = all)
- `temperature=0.3`: generation temperature
- `max_tokens=512`: maximum response tokens

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
pip install -e ".[search,chat-ollama]"

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

The provider uses `requests` to call the Ollama HTTP API. It does not use the Ollama Python package.
The availability checker pings this endpoint first. If unavailable, rsi tries the other local provider.

## OpenAI-Compatible (Cloud / API)

Works with OpenAI, vLLM, LM Studio, and any server that implements the OpenAI chat completions API.

### Setup with OpenAI

```bash
# 1. Install the provider
pip install -e ".[search,chat-openai]"

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

## Context budgets

Chat reserves the configured response tokens before packing complete source passages and conversation history.
Oldest history turns are removed first. Sources that do not fit are omitted from both the prompt and citations.
If no source fits, RSI returns an explanation without calling the model.
Rebuild legacy indexes into passages to use long-thread evidence within small context windows.

Local GGUF models use their chat template and tokenizer for the count.
OpenAI-compatible and Ollama templates use a conservative byte-based estimate with a framing reserve.
Unusual custom server templates can exceed that reserve; use a server-specific counter before enabling them.
The Ollama request sets `num_ctx` to the provider's advertised context window.

## Auto-Fallback

When chat starts, rsi checks the configured provider first. If unavailable, it tries these alternatives:

1. **llama-cpp:** checks that `llama-cpp-python` is importable and a GGUF file exists.
2. **ollama:** pings `http://localhost:11434/api/tags`.
3. **openai:** considered only when cloud fallback is explicitly allowed. Checks the SDK and `OPENAI_API_KEY`.

The CLI and UI display the resolved destination and model before generation.
MCP tool discovery advertises the configured destination and whether cloud fallback is permitted.
The MCP server logs the resolved destination before each chat call. Clients may not display those logs.
MCP operators must communicate the configured destinations to users before enabling chat.

### Example Fallback Scenarios

| Configured | What happens | Falls back to |
|-----------|-------------|---------------|
| `llama-cpp` | No GGUF file in `~/.rsi/models/` | `ollama` (if running) |
| `ollama` | Ollama server not running | `llama-cpp` (if GGUF available) |
| `openai` | No API key set | `llama-cpp` (if GGUF available) |
| `llama-cpp` | Both local providers unavailable | Error, unless cloud fallback is permitted |

### Permitting Cloud Fallback

Cloud fallback is disabled by default, including when an API key already exists.
To permit archive context and conversation history to reach an OpenAI-compatible endpoint, set:

```toml
[llm]
allow_cloud_fallback = true
```

Or set `RSI_ALLOW_CLOUD_FALLBACK=true`. Only `true` and `false` are accepted; other values raise an error.
Selecting `openai` directly also permits that endpoint. `OPENAI_BASE_URL` sets its destination; the default is `https://api.openai.com/v1`.
The `--provider` flag changes the selected provider. Local fallback still applies if that provider is unavailable.
This setting controls chat only. An explicit OpenAI embedding provider separately sends indexed text and search queries to its API.
