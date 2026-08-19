# Padwan LLM

Unified client for OpenAI, Gemini, Mistral, Grok, and Anthropic APIs. Supports also OpenAI-compatible endpoints.

## Why

Most LLM client libraries pull in heavy dependencies (pydantic, httpx) and lock you into a single provider's SDK. Padwan LLM takes a different approach:

- **Single runtime dependency** — only [niquests](https://github.com/jawah/niquests), no pydantic, no httpx. Zero overhead beyond the HTTP layer.
- **TypedDict-only** — all request/response types are plain `TypedDict`s, no validation framework required. No runtime cost, full editor support.
- **Multi-provider, extensible** — supports the major providers (OpenAI, Gemini, Mistral, Grok, Anthropic) with a shared base class that makes adding new ones straightforward.

## Features

- **Unified interface** - Single API for multiple LLM providers
- **Async-first** - Built on async/await for high performance
- **HTTP/2 and HTTP/3** - Automatic protocol negotiation via [niquests](https://github.com/jawah/niquests)
- **Fully typed** - Complete type hints with Python 3.13+ generics
- **Streaming support** - Real-time token streaming for all providers
- **Conversation management** - Built-in conversation history handling
- **Agentic loop** - `AgentSession` drives multi-turn conversations with tool dispatch, parallel execution, approval hooks, and snapshot persistence
- **MCP support** - Streamable HTTP and stdio transports for [Model Context Protocol](https://modelcontextprotocol.io/) tool servers
- **Thinking models** - Stream Gemini and Claude thought tokens separately via an `on_thought` callback
- **Realtime voice** - Speech-to-speech WebSocket sessions (OpenAI Realtime, Gemini Live, Grok Voice) via `RealtimeClient`

## Supported Providers

| Provider | Chat | Streaming | Batch | Realtime | Transcription | Embeddings |
|----------|------|-----------|-------|----------|---------------|------------|
| OpenAI   | ✅   | ✅        | ✅    | ✅       | ❌            | ❌         |
| Gemini   | ✅   | ✅        | ✅    | ✅       | ❌            | ❌         |
| Mistral  | ✅   | ✅        | ❌    | ❌       | ✅            | ✅         |
| Grok     | ✅   | ✅        | ✅    | ✅       | ❌            | ❌         |
| Anthropic | ✅  | ✅        | ❌    | ❌       | ❌            | ❌         |
| [OpenAI-Compatible](clients/openai-compatible.md) | ✅ | ✅ | ➕ | ➕ | ➕ | ➕ |

## Quick Example

```bash
export OPENAI_API_KEY="sk-..."
uv run padwan-llm "Hello!" -m gpt-4o-mini
```

```python
from padwan_llm import LLMClient

async with LLMClient(model="gpt-4o") as client:
    response, usage = await client.complete_chat(
        [{"role": "user", "content": "Hello!"}]
    )
    print(response["content"])
```

## Installation

```bash
pip install padwan-llm
```

Or with uv:

```bash
uv add padwan-llm
```

## Agentic loop

`AgentSession` wraps a conversation with the loop that calls the LLM, dispatches any tool calls, feeds the results back, 
and repeats until the model produces a plain text answer.
It accepts individual `McpTool` instances and whole `McpTransport` servers in the same list, transports are entered as part of the session lifecycle:

```python
from padwan_llm import AgentSession, LLMClient, McpStdio

async with AgentSession(
    client=LLMClient(model="gpt-4o"),
    mcp_tools=[McpStdio(command="uvx", args=["weather-mcp"])],
    system="You have access to weather tools.",
) as session:
    text = await session.send("What is the weather in Paris?")
```

See the [Agents page](agents.md) for approval hooks, parallel execution, and snapshot persistence.

## MCP (Model Context Protocol)

Connect to MCP tool servers over streamable HTTP or stdio:

```python
from padwan_llm import McpStreamable, McpStdio

async with McpStreamable(url="https://mcp.example.com/mcp", token="sk-...") as mcp:
    result = await mcp.tools[0].handler({"query": "test"})
```

See the [MCP page](mcp.md) for architecture diagrams, the dual-channel design of `McpStreamable`, the stdio reader model, and the reconnection flow.

## CLI / TUI

The interactive CLI/TUI is available as a separate package: `padwan-cli`.

```bash
# One-shot prompt
uvx padwan-cli "Explain Python decorators" -m gemini-2.5-flash

# Interactive chat
uvx padwan-cli chat -m gpt-4o
```
