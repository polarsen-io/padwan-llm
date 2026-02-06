# Getting Started

## Installation

=== "pip"

    ```bash
    pip install polarsen-llm
    ```

=== "uv"

    ```bash
    uv add polarsen-llm
    ```

## Basic Usage

### Creating a Client

```python
from polarsen_llm import LLMClient

# Using context manager (recommended)
async with LLMClient("gpt-4o") as client:
    response = await client.chat("Hello!")

# Or manually manage the client
client = LLMClient("gemini-2.0-flash")
response = await client.chat("Hello!")
await client.close()
```

### Supported Models

The provider is auto-detected from the model name:

- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini`, ...
- **Gemini**: `gemini-2.0-flash`, `gemini-1.5-pro`, ...
- **Mistral**: `mistral-large-latest`, `mistral-small-latest`, ...
- **Grok**: `grok-2`, `grok-2-mini`, ...

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
async with LLMClient("gpt-4o") as client:
    response = await client.chat("Hello!")
```

## Streaming

All clients support streaming responses:

```python
async with LLMClient("gpt-4o") as client:
    async for chunk in client.stream("Tell me a story"):
        print(chunk.content, end="", flush=True)
```

## Conversations

Maintain conversation history across multiple messages:

```python
from polarsen_llm import LLMClient, Conversation

conv = Conversation()
conv.add_user("What is Python?")

async with LLMClient("gpt-4o") as client:
    response = await client.chat(conv)
    conv.add_assistant(response.content)

    conv.add_user("What are its main features?")
    response = await client.chat(conv)
```