# Padwan LLM

Unified async client for OpenAI, Gemini, Mistral, and Grok APIs.

## Why

Most LLM client libraries pull in heavy dependencies (pydantic, httpx) and lock you into a single provider's SDK. Padwan LLM takes a different approach:

- **Single runtime dependency** — only [niquests](https://github.com/jawah/niquests), no pydantic, no httpx. Zero overhead beyond the HTTP layer.
- **TypedDict-only** — all request/response types are plain `TypedDict`s, no validation framework required. No runtime cost, full editor support.
- **Multi-provider, extensible** — supports the major providers (OpenAI, Gemini, Mistral, Grok) with a shared base class that makes adding new ones straightforward.

## Features

- **Unified interface** - Single API for multiple LLM providers
- **Async-first** - Built on async/await for high performance
- **HTTP/2 and HTTP/3** - Automatic protocol negotiation via [niquests](https://github.com/jawah/niquests)
- **Fully typed** - Complete type hints with Python 3.14+ generics
- **Streaming support** - Real-time token streaming for all providers
- **Conversation management** - Built-in conversation history handling

## Supported Providers

| Provider | Chat | Streaming | Batch | Transcription | Embeddings |
|----------|------|-----------|-------|---------------|------------|
| OpenAI   | ✅   | ✅        | ✅    | ❌            | ❌         |
| Gemini   | ✅   | ✅        | ✅    | ❌            | ❌         |
| Mistral  | ✅   | ✅        | ❌    | ✅            | ✅         |
| Grok     | ✅   | ✅        | ✅    | ❌            | ❌         |
| [OpenAI-Compatible](clients/openai-compatible.md) | ✅ | ✅ | ➕ | ➕ | ➕ |

## Quick Example

```bash
export OPENAI_API_KEY="sk-..."
uv run padwan-llm "Hello!" -m gpt-4o-mini
```

```python
from padwan_llm import LLMClient

async with LLMClient("gpt-4o") as client:
    response = await client.chat("Hello, world!")
    print(response.content)
```

## Installation

```bash
pip install padwan-llm
```

Or with uv:

```bash
uv add padwan-llm
```

## CLI / TUI

The interactive CLI/TUI is available as a separate package: `padwan-cli`.

```bash
# One-shot prompt
uvx padwan-cli "Explain Python decorators" -m gemini-2.0-flash

# Interactive chat
uvx padwan-cli chat -m gpt-4o
```
