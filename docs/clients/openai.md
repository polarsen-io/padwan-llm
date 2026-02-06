# OpenAI Client

The OpenAI client provides access to GPT models through the OpenAI API.

## Configuration

```python
from src.openai import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    model="gpt-4o",    # default model
)
```

## Available Models

- `gpt-4o` - Most capable model
- `gpt-4o-mini` - Faster, cheaper variant
- `gpt-4-turbo` - Previous generation
- `gpt-3.5-turbo` - Legacy model

## Usage

### Basic Chat

```python
async with OpenAIClient() as client:
    response = await client.chat("Hello!")
    print(response.content)
```

### Streaming

```python
async with OpenAIClient() as client:
    async for chunk in client.stream("Tell me a story"):
        print(chunk.content, end="")
```

### With System Prompt

```python
from src import Conversation

conv = Conversation(system="You are a helpful assistant.")
conv.add_user("Hello!")

async with OpenAIClient() as client:
    response = await client.chat(conv)
```

## Response Format

```python
@dataclass
class ChatResponse:
    content: str
    model: str
    usage: Usage | None
    finish_reason: str | None
```