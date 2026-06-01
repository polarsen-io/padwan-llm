## 0.7.1 (2026-06-01)

Maintenance release: a Mistral model-ID correction, regenerated provider types, and a batch of improvements to the weekly model-drift automation.

### Fixes

- **Mistral**: correct the Mistral Medium 3.5 model ID from `mistral-medium-3.5` to `mistral-medium-3-5`, the form Mistral's docs and API use (#15).
- Regenerate OpenAI and Mistral OpenAPI TypedDicts against the latest SDKs (#15).

### Drift automation (tooling)

- Surface **tracked-but-deprecated** models: the drift report now reads each provider's `deprecation` field and warns before a model's retirement date, instead of only reacting once it disappears from the API (#17).
- Sign the weekly automation bot's commits with an SSH signing key so they show as **Verified** (#16).
- Emit stdlib TypedDicts via `--no-use-closed-typed-dict` during type regeneration (#14).
- Fetch the automation branch before pushing to avoid stale-ref push failures (#12).

**Full Changelog**: https://github.com/polarsen-io/padwan-llm/compare/0.7.0...0.7.1

## 0.7.0 (2026-05-15)

### Features

- **Model drift workflow** — weekly Monday cron (`model-drift.yml`) that bumps the `llms` dependency group, regenerates OpenAI/Mistral OpenAPI TypedDicts, and opens or refreshes an `automation/model-update` PR with a drift report. Companion scripts in `bin/drift/`: `check_model_drift.py` (provider model diff), `refresh-llms.sh` (local end-to-end refresh), `pr.sh` (open-or-refresh/close-stale PR helpers).
- **New model IDs** across all providers (first automated drift run):
  - OpenAI: `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.4-nano`
  - Gemini: `gemini-3-pro-preview`, `gemini-3.1-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-pro-latest`
  - Mistral: `devstral-medium-latest`, `ministral-14b-latest`, `mistral-medium`, `mistral-medium-3`, `mistral-medium-3.5`, `mistral-tiny-latest`, `codestral-embed`, `voxtral-mini-realtime-latest`, `voxtral-mini-tts-latest`, `voxtral-small-latest`
  - Grok: `grok-3-fast`, `grok-3-mini-fast`, `grok-4-1-fast`, `grok-4.20`, `grok-4.20-non-reasoning`, `grok-4.20-reasoning`, `grok-4.3`, `grok-code-fast`, and `-latest` aliases for Grok 3/4 families

### Chore

- Bump SDK floors: `openai>=2.36.0`, `google-genai>=2.1.0`, `xai-sdk>=1.12.2`, `mcp>=1.27.1`
- Move `google-genai`, `mcp`, `openai`, `xai-sdk` into a dedicated `[dependency-groups.llms]` (enumerable by the drift workflow)
- Scope pyright to `padwan_llm` and `tests` to avoid OOM on transitive SDK sources

## 0.6.0 (2026-05-02)

### Feat

- **mcp**: add `McpStreamable` (HTTP) and `McpStdio` (subprocess) transports implementing the [MCP 2025-11-25 spec](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports), with tool discovery, bearer auth, session management, ping, progress notifications, request cancellation, and in-place tool refresh on `notifications/tools/list_changed`
- **agent** *(alpha)*: add `AgentSession` — multi-turn agentic loop with streaming, sequential/parallel tool execution, approval hooks, error handlers, tool-result truncation, and snapshot persistence via `ConversationStore`. Per-round tool refresh picks up MCP `tools/list_changed` without restarting the session. Automatic tool namespacing on collision via `auto_prefix`, with startup rollback on partial initialization failure
- **thoughts**: unified `on_thought` callback across Gemini, Grok, and Mistral for streaming model reasoning tokens separately from the final answer
- **gemini**: thinking models support (`thinkingConfig`, `thinkingBudget`) with thought/answer separation in both streaming and non-streaming paths
- support Python 3.13 (`requires-python >= 3.13`)

### Refactor

- replace manual SSE line parsing in OpenAI and Gemini streaming with niquests native SSE extension (`resp.extension`)
- Use `orjson` when available
- **docs**: migrate from mkdocs-material to [zensical](https://github.com/polarsen-io/zensical)

### Fix

- thought tokens leaking into `complete_chat` text output on Gemini
- empty SSE `data:` frames (keep-alives) treated as malformed JSON
- OpenAI/Grok SSE responses missing `charset` in `Content-Type` causing `TypeError` on stream iteration

### Dependencies

- bump niquests to `>=3.18.5`
- bump urllib3-future to `>=2.19.905`

### Agent alpha limitations

The `AgentSession` API is functional but considered alpha — the interface may change in future releases. Known limitations:

- No mid-stream approval — `approve_tool` runs after all tool calls for a round are known, not as they arrive
- Tool results are string-only — MCP image/resource content blocks are JSON-serialized
- No per-call cancellation — you can cancel the whole `send()`/`stream()` task, but not an individual in-flight tool call
