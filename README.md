<p align="center">
  <img src="docs/static/logo-hood.png" alt="Padwan LLM" width="120">
</p>

<h1 align="center">Padwan LLM</h1>

Lightweight, unified async client for OpenAI, Gemini, Mistral, Grok, Anthropic, and any OpenAI-compatible API.
Single runtime dependency ([niquests](https://github.com/jawah/niquests)), automatic HTTP/2 and HTTP/3 negotiation.

For the full interactive CLI/TUI, use the separate [`padwan-cli`](https://github.com/polarsen-io/padwan-cli) package.

<img alt="Chat demo" src="https://github.com/polarsen-io/padwan-cli/raw/master/docs/static/chat.gif" width="500"/>

## Installation

```bash
pip install padwan-llm
```

## Library Usage

### One-shot chat

```python
from padwan_llm import LLMClient

async with LLMClient(model="gpt-4o") as client:
    response, usage = await client.complete_chat(
        [{"role": "user", "content": "Hello!"}]
    )
    print(response["content"])
```

### Streaming with `ConversationState`

```python
from padwan_llm import LLMClient, ConversationState

state = ConversationState(system="You are a concise assistant.")

async with LLMClient(model="gpt-4o") as client:
    state.add_user_message("What's Python?")

    stream = client.stream_chat(state.messages)
    chunks: list[str] = []
    async for text in stream:
        print(text, end="", flush=True)
        chunks.append(text)

    state.add_assistant_message("".join(chunks))
    if stream.usage:
        state.accumulate_usage(stream.usage)
```

### Agentic loop with `AgentSession`

`AgentSession` drives a multi-turn conversation that can dispatch tool calls on each
round, feed the results back, and repeat until the model returns a plain text answer.
The `mcp_tools` list accepts both individual `McpTool` instances and whole
`McpTransport` servers — transports are entered as part of the session lifecycle:

```python
from padwan_llm import AgentSession, LLMClient, McpStdio

async with AgentSession(
    client=LLMClient(model="gpt-4o"),
    mcp_tools=[McpStdio(command="uvx", args=["my-mcp-server"])],
    system="You have access to tools. Use them when helpful.",
) as session:
    async for chunk in session.stream("What's the weather in Paris?"):
        print(chunk, end="", flush=True)

    # Or collect the full response in one call:
    text = await session.send("And in London?")
```

`AgentSession` supports sequential or parallel tool execution, approval hooks,
per-tool error handlers, and optional snapshot persistence via a
`ConversationStore` protocol — see [docs/agents.md](docs/agents.md).

### MCP (Model Context Protocol)

Both streamable-HTTP and stdio MCP transports are built in:

```python
from padwan_llm import McpStreamable, McpStdio

# Remote MCP server over HTTP (with optional bearer token)
async with McpStreamable(url="https://mcp.example.com/mcp", token="sk-...") as mcp:
    for tool in mcp.tools:
        print(tool.name, tool.description)

# Local subprocess
async with McpStdio(command="uvx", args=["my-mcp-server"]) as mcp:
    result = await mcp.tools[0].handler({"query": "hello"})
```

See [docs/mcp.md](docs/mcp.md) for the full feature matrix and architecture.

### Gemini thinking models

Gemini's reasoning models can stream their internal thought tokens separately from
the final answer. Wire an `on_thought` callback to receive them:

```python
from padwan_llm import GeminiClient

thoughts: list[str] = []
async with GeminiClient(
    model="gemini-2.5-flash",
    on_thought=thoughts.append,
    thinking_config={"thinkingBudget": 2048, "includeThoughts": True},
) as client:
    stream = client.stream_chat([{"role": "user", "content": "What is 7 * 8?"}])
    async for chunk in stream:
        print(chunk, end="")

print("\n---\nReasoning:", "".join(thoughts))
```

### Realtime speech-to-speech

`RealtimeClient` opens a bidirectional voice session over a WebSocket and yields
the live connection: stream microphone audio in, receive model audio and
transcripts back. OpenAI (`gpt-realtime`), Gemini Live, and Grok Voice are
supported, dispatched by model name. Requires the `realtime` extra
(`pip install "padwan-llm[realtime]"`):

```python
from padwan_llm import RealtimeClient

async with RealtimeClient(instructions="Answer briefly.", voice="marin") as conn:
    await conn.append_audio(pcm16_chunk)  # mono PCM16 microphone audio
    async for event in conn:
        if audio := conn.audio_delta_bytes(event):
            playback.write(audio)
```

Server-side VAD drives turn-taking by default; pass `turn_detection=NO_TURN_DETECTION`
for manual push-to-talk. See the realtime sections of
[docs/clients/openai.md](docs/clients/openai.md),
[docs/clients/gemini.md](docs/clients/gemini.md), and
[docs/clients/grok.md](docs/clients/grok.md).

## One-Shot Command

```bash
export OPENAI_API_KEY=...

padwan-llm "Hello!" -m gpt-4o-mini

# Or without installing:
uvx padwan-llm "Hello!" -m gpt-4o-mini
```

## Supported Models

Auto-detected providers: **OpenAI**, **Gemini**, **Mistral**, **Grok**, **Anthropic** (`claude-*`).

Any OpenAI-compatible API (Groq, Together AI, Ollama, vLLM, ...) is supported via `OpenAIClient` with a custom `base_url`.

## Testing

Unit tests run by default (no API keys needed):

```bash
uv run pytest
```

E2e tests require API keys. Create a `.env` file or pass one with `--env-file`:

```bash
uv run pytest tests/e2e/ -m e2e
uv run pytest tests/e2e/ -m e2e --env-file path/to/.env
```

Tests for providers whose API key is missing are automatically skipped.

## Environment Variables

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROK_API_KEY=...
ANTHROPIC_API_KEY=...
```

### Unified gateway (one URL + one token)

Aggregators that expose OSS variants of many model families behind a single
OpenAI-compatible endpoint and token are supported with two env vars — every
model then routes through that gateway, with no per-provider keys or per-call
overrides:

```bash
PADWAN_BASE_URL=https://your-gateway.example.com/v1/
PADWAN_API_KEY=...
```

```python
# Names that would normally route to a native client (gemini-*, mistral-*, …)
# go through the gateway as OpenAI-compatible instead.
async with LLMClient(model="gemini-2.5-flash") as client:
    response, usage = await client.complete_chat([{"role": "user", "content": "Hi!"}])
```

Precedence is explicit `base_url`/`api_key` args → `PADWAN_*` → native
per-provider env vars. Passing an explicit `base_url` disables gateway mode and
restores native provider routing.
