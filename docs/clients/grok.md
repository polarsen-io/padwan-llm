# Grok Client

The Grok client provides access to xAI's Grok models.

## Configuration

```python
from src.grok import GrokClient

client = GrokClient(
    api_key="...",      # or set GROK_API_KEY env var
    model="grok-2",     # default model
)
```

## Available Models

- `grok-2` - Latest Grok model
- `grok-2-mini` - Faster, smaller variant

## Usage

### Basic Chat

```python
async with GrokClient() as client:
    response = await client.chat("Hello!")
    print(response.content)
```

### Streaming

```python
async with GrokClient() as client:
    async for chunk in client.stream("Tell me a story"):
        print(chunk.content, end="")
```

### With System Prompt

```python
from src import Conversation

conv = Conversation(system="You are a helpful assistant.")
conv.add_user("Hello!")

async with GrokClient() as client:
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