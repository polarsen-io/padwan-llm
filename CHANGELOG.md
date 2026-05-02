## 0.6.0 (2026-05-02)

### Feat

- **mcp**: add `McpStreamable` (HTTP) and `McpStdio` (subprocess) transports implementing the [MCP 2025-11-25 spec](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports), with tool discovery, bearer auth, session management, ping, progress notifications, request cancellation, and in-place tool refresh on `notifications/tools/list_changed`
- **agent** *(alpha)*: add `AgentSession` — multi-turn agentic loop with streaming, sequential/parallel tool execution, approval hooks, error handlers, tool-result truncation, and snapshot persistence via `ConversationStore`. Per-round tool refresh picks up MCP `tools/list_changed` without restarting the session. Automatic tool namespacing on collision via `auto_prefix`, with startup rollback on partial initialization failure
- **thoughts**: unified `on_thought` callback across Gemini, Grok, and Mistral for streaming model reasoning tokens separately from the final answer
- **gemini**: thinking models support (`thinkingConfig`, `thinkingBudget`) with thought/answer separation in both streaming and non-streaming paths
- support Python 3.13 (`requires-python >= 3.13`)

### Refactor

- replace manual SSE line parsing in OpenAI and Gemini streaming with niquests native SSE extension (`resp.extension`)
- add `dumps` to `_json.py` (orjson-backed when available), replace all `json.dumps` / `json.JSONDecodeError` across the codebase
- cache `AgentSession._build_round_dispatch()` via identity-based fingerprinting — skip rebuild when tool list is unchanged
- `AgentSession.load()`: replace `**kwargs` with explicit typed parameters
- **docs**: migrate from mkdocs-material to [zensical](https://github.com/polarsen-io/zensical); switch GitHub Pages deploy to official `actions/deploy-pages`

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

### Known issue

niquests' SSE extension drops events when multiple SSE messages arrive in a single HTTP/2 DATA frame (e.g. OpenAI streaming tool-call arguments). Fix submitted upstream: jawah/urllib3.future#344. Until merged and released, streaming tool calls may produce garbled arguments on affected providers.

## 0.5.1 (2026-03-03)

### Fix

- update Grok model definitions to match current xAI API
