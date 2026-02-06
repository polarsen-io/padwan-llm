# Polarsen LLM

Unified async client for OpenAI, Gemini, Mistral, and Grok APIs.

## Features

- **Unified interface** - Single API for multiple LLM providers
- **Async-first** - Built on async/await for high performance
- **Fully typed** - Complete type hints with Python 3.14+ generics
- **Streaming support** - Real-time token streaming for all providers
- **Conversation management** - Built-in conversation history handling

## Supported Providers

| Provider | Chat | Streaming | Batch |
|----------|------|-----------|-------|
| OpenAI   | ✅   | ✅        | ❌    |
| Gemini   | ✅   | ✅        | ✅    |
| Mistral  | ✅   | ✅        | ❌    |
| Grok     | ✅   | ✅        | ❌    |

## Quick Example

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