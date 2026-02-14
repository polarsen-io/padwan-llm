# Polarsen LLM

Unified async client for OpenAI, Gemini, Mistral, and Grok APIs.

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
| OpenAI   | ✅   | ✅        | ❌    | ❌            | ❌         |
| Gemini   | ✅   | ✅        | ✅    | ❌            | ❌         |
| Mistral  | ✅   | ✅        | ❌    | ✅            | ✅         |
| Grok     | ✅   | ✅        | ❌    | ❌            | ❌         |

## Quick Example

```bash
export OPENAI_API_KEY="sk-..."
uv run padwan-llm "Hello!" -m gpt-4o-mini
```

```python
from polarsen_llm import LLMClient

async with LLMClient("gpt-4o") as client:
    response = await client.chat("Hello, world!")
    print(response.content)
```

## Installation

```bash
pip install polarsen-llm
```

Or with uv:

```bash
uv add polarsen-llm
```

## CLI / TUI

The interactive CLI/TUI is available as a separate package: `padwan-cli`.

```bash
# One-shot prompt
uvx padwan-cli "Explain Python decorators" -m gemini-2.0-flash

# Interactive chat
uvx padwan-cli chat -m gpt-4o
```