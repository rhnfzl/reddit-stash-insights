# Release safety result

Task 1 is implemented without commits or corpus access.

- All three search modes use one escaped SQL literal predicate. Actual LanceDB tests cover apostrophes, expression injection, backslashes, and Unicode.
- Automatic cloud fallback requires keyword-only `allow_cloud_fallback=True`. Direct cloud selection remains supported. Local fallback remains available.
- `Settings.allow_cloud_fallback` defaults to false. TOML requires a boolean. `RSI_ALLOW_CLOUD_FALLBACK` accepts only true or false (case-insensitive, surrounding whitespace ignored).
- CLI, UI, and MCP pass this setting. CLI and UI display the resolved destination before generation. MCP discovery describes configured destinations and cloud fallback consent; the server logs the resolved destination before generation.
- Destination displays remove URL user information, query parameters, and fragments.
- Destination names normalize case and whitespace consistently with provider selection. A regression first reproduced misleading in-process reporting for ` OpEnAI `.
- Package/CLI claims describe implemented search and chat. Removed graph/analytics extras and unused UI plotly dependency. Ollama extra now installs requests.
- README/provider setup uses quoted source-install extras. Configuration docs include fallback consent. The root agent corrected the public repository description.
- Scanned all tracked documentation for unquoted extras and unsupported graph/analytics promises. Corrected the remaining MCP setup command and its API-key-only cloud setup guidance.

Observed regression failures before implementation:

- Apostrophe filter values caused SQL parse errors in every mode.
- Injected `ordinary' OR 1=1 --` returned all five fixture rows in every mode.
- Failed local providers selected OpenAI without consent.
- Consent settings were missing and invalid boolean values were accepted.
- New destination helper was absent; CLI did not pass fallback consent.

Verification:

```text
.venv/bin/python -m unittest tests.test_vector_store tests.test_chat_providers tests.test_config tests.test_chat_engine tests.test_cli_chat tests.test_mcp_http -q
Ran 94 tests in 1.224s, OK

.venv/bin/python -m ruff check <all touched source and test files>
All checks passed!

.venv/bin/python -m mypy src/rsi/indexer/vector_store.py src/rsi/chat/providers/availability.py src/rsi/config.py src/rsi/cli.py src/rsi/ui/chat.py src/rsi/mcp_server.py
Success: no issues found in 6 source files

git diff --check
Passed
```

Project runtime: `.venv/bin/python`, LanceDB 0.41.0 from this project's `.venv/lib/python3.14/site-packages/lancedb/__init__.py`.
LanceDB emits an existing `create_fts_index` deprecation warning. Provider/HTTP rejection logs are expected test paths.

`ab_test_rag.py` and `tests/fixtures/rag_test_questions.json` remain ignored and untracked.

Not checked: live provider requests, rendered UI, isolated extras installs, and publication. The root agent owns full-suite and release verification.
Risk: MCP clients may hide server logs. Operators must disclose configured destinations before enabling chat. Chat fallback consent does not control an explicitly configured hosted embedding provider.
