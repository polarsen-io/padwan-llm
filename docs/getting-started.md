# Getting Started

## Installation

=== "pip"

    ```bash
    pip install padwan-llm
    ```

=== "uv"

    ```bash
    uv add padwan-llm
    ```

## Basic Usage

### Creating a Client

```python
from padwan_llm import LLMClient
from padwan_llm.conversation import Message

# Using context manager (recommended)
async with LLMClient("gpt-4o") as client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])

# Or manually manage the client
client = LLMClient("gemini-2.0-flash")
async with client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
```

### Supported Models

The provider is auto-detected from the model name: **OpenAI**, **Gemini**, **Mistral**, **Grok**.

For other providers, use `OpenAIClient` directly:

```python
from padwan_llm import OpenAIClient

async with OpenAIClient(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1/",
    api_key="gsk-...",
) as client:
    text, usage = await client.complete_chat([
        {"role": "user", "content": "Hello!"}
    ])
```

### Environment Variables

Each provider looks for its API key in environment variables:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI   | `OPENAI_API_KEY`    |
| Gemini   | `GEMINI_API_KEY`    |
| Mistral  | `MISTRAL_API_KEY`   |
| Grok     | `GROK_API_KEY`      |

```python
# No need to pass api_key if environment variable is set
from padwan_llm import LLMClient
from padwan_llm.conversation import Message

async with LLMClient("gpt-4o") as client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
```

## Streaming

All clients support streaming responses:

```python
from padwan_llm.conversation import Message

async with LLMClient("gpt-4o") as client:
    stream = client.stream_chat([
        Message(role="user", content="Tell me a story")
    ])
    async for chunk in stream:
        print(chunk, end="", flush=True)

    if stream.usage:
        print(f"\nTokens used: {stream.usage['total']}")
```

## Conversations

Maintain conversation history across multiple messages:

```python
from padwan_llm import LLMClient, ConversationState

state = ConversationState(system="You are a helpful assistant.")
state.add_user_message("What is Python?")

async with LLMClient("gpt-4o") as client:
    stream = client.stream_chat(state.messages)
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    state.add_assistant_message("".join(chunks))
    if stream.usage:
        state.accumulate_usage(stream.usage)

    state.add_user_message("What are its main features?")
    text, usage = await client.complete_chat(state.messages)
    state.add_assistant_message(text)
    state.accumulate_usage(usage)
```
