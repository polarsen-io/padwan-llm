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
