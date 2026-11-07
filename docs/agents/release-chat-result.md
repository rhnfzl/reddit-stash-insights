# Chat evidence packing result

Chat now keeps complete retrieved passages and cites only passages that fit the model input budget. Facts after character 500 remain available.

## Interfaces and behavior

- `LLMProvider` now requires `max_output_tokens` and `count_tokens(messages)` beside `context_window`.
- `build_messages(query, docs, history, system=SYSTEM_PROMPT, *, input_budget, count_tokens)` returns `(messages, included_sources)`.
- The engine subtracts the provider output limit from its context window before packing input.
- Counting includes system instructions, the question, evidence metadata, passage text, history, and message framing.
- Oldest complete history turns leave first. Ranked passages remain whole. Oversized passages are skipped so smaller later passages can fit.
- If system instructions and the question alone exceed the input budget, packing raises a clear `ValueError`.
- Empty retrieval or no fitting passages returns an explicit abstention without calling the model.
- Streaming and complete answers use the same packer, source list, and history source list.
- Archive text is quoted as JSON and marked untrusted. System instructions prohibit following archive instructions. This is not a jailbreak prevention guarantee.
- CLI, Streamlit, and MCP display evidence ID, chunk ID, evidence permalink, file path, and body character offsets when provided.

## Provider counts and limits

The local llama-cpp provider uses one formatter for counting and generation. A GGUF chat template uses `Jinja2ChatFormatter` with the model BOS/EOS strings and EOS stopping IDs. A GGUF without that template uses the existing llama-2 fallback formatter. The loaded model tokenizer counts the rendered prompt, including special tokens and the assistant prefix. A missing BOS ID remains empty. A templated GGUF without an EOS ID fails clearly before generation. Root added this regression after the worker report.

Runtime inspection used `.venv/bin/python`. Installed llama-cpp-python is `0.3.36`, resolved at `/Users/rehan-8v/projects/personal/reddit-stash-insights/.venv/lib/python3.14/site-packages/llama_cpp/__init__.py`. Formatter behavior was read from that runtime's `llama_chat_format.py`, including `add_bos=not rendered.added_special` and `special=True`.

Ollama and OpenAI-compatible APIs do not expose their exact tokenizers or server chat templates through the existing provider interface. Their conservative fallback counts every UTF-8 byte as one token, adds the role byte count, reserves 128 tokens per message, and adds 256 tokens for reply framing. This overcounts normal byte-level text models. A custom template with greater framing overhead can exceed this estimate. A provider-specific counter is required before claiming support for such a template. No dependency or tokenizer download was added.

Ollama requests now send `num_ctx=context_window` and `num_predict=max_output_tokens`. This prevents the server defaulting to a smaller context. All constructors reject non-positive output limits and output limits at least as large as the configured context window.

## Verification

TDD failures were observed before each implementation group:

- Engine/prompt regressions: 5 failed assertions and 7 missing-interface errors before packing existed.
- Provider regressions: 3 failed assertions and 4 missing-interface/payload errors before provider budgets existed.
- CLI/MCP citation regressions: 2 failed assertions before evidence locations were displayed.
- Streamlit citation regression: 1 failed assertion before evidence locations were displayed.
- Missing BOS regression: 1 failed assertion before negative token IDs were excluded.

Final scoped check:

```text
.venv/bin/python -m unittest tests.test_chat_providers tests.test_chat_engine tests.test_cli_chat tests.test_mcp_http tests.test_ui_chat_sources -q
Ran 94 tests in 0.388s
OK
```

This covers late evidence, Unicode, history eviction, output reservation, oversized questions, skipped passage citations, empty retrieval, stream parity, protocol fixtures, Ollama context requests, local tokenizer arguments, and evidence identity in CLI/MCP/Streamlit. The Streamlit AppTest emits expected bare-mode ScriptRunContext warnings.

Touched-source and touched-test ruff checks passed. `.venv/bin/mypy src/rsi/` passed for 30 source files.

No model was loaded by this worker. Root owns the actual cached-Qwen synthetic smoke and full release checks. No personal corpus or hosted generation was used. No commits or dependency changes were made by this worker.
