# OpenAI-Compatible APIs

`OpenAIClient` can be used directly with any API that implements the OpenAI chat completions interface. Use it to connect to providers like Groq, Together AI, Fireworks, Ollama, vLLM, LiteLLM, or any other OpenAI-compatible endpoint.

The named provider clients (`MistralClient`, `GrokClient`) inherit from this class.

## Configuration

```python
from padwan_llm import OpenAIClient

client = OpenAIClient(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1/",
    api_key="gsk-...",
)
```

Both `base_url` and `api_key` are required when targeting a non-OpenAI endpoint.

## Usage

### Basic Chat

```python
async with OpenAIClient(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1/",
    api_key="gsk-...",
) as client:
    text, usage = await client.complete_chat([
        {"role": "user", "content": "Hello!"}
    ])
    print(text)
```

### Streaming

```python
async with OpenAIClient(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1/",
    api_key="gsk-...",
) as client:
    stream = client.stream_chat([
        {"role": "user", "content": "Tell me a story"}
    ])
    async for chunk in stream:
        print(chunk, end="", flush=True)

    if stream.usage:
        print(f"\nTokens used: {stream.usage['total']}")
```

## Examples

### Ollama (local)

```python
async with OpenAIClient(
    model="llama3",
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # Ollama doesn't require a real key
) as client:
    text, usage = await client.complete_chat([
        {"role": "user", "content": "Hello!"}
    ])
```

### Together AI

```python
async with OpenAIClient(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    base_url="https://api.together.xyz/v1/",
    api_key="...",
) as client:
    text, usage = await client.complete_chat([
        {"role": "user", "content": "Hello!"}
    ])
```

### vLLM

```python
async with OpenAIClient(
    model="meta-llama/Llama-3-8b-chat-hf",
    base_url="http://localhost:8000/v1/",
    api_key="token-abc123",
) as client:
    text, usage = await client.complete_chat([
        {"role": "user", "content": "Hello!"}
    ])
```

## Subclassing

To create a reusable client for a specific provider, subclass `OpenAIClient`:

```python
import dataclasses
import os
from typing import ClassVar
from padwan_llm import OpenAIClient
from padwan_llm.errors import LLMError

@dataclasses.dataclass
class GroqClient(OpenAIClient):
    provider: ClassVar[str] = "groq"
    model: str | None = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1/"

    def _get_default_api_key(self) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError(self.provider, "GROQ_API_KEY not set")
        return api_key
```