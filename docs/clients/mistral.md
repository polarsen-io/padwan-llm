# Mistral Client

The Mistral client provides access to Mistral AI models.

## Configuration

```python
from src.mistral import MistralClient

client = MistralClient(
    api_key="...",           # or set MISTRAL_API_KEY env var
    model="mistral-large-latest",  # default model
)
```

## Available Models

- `mistral-large-latest` - Most capable model
- `mistral-medium-latest` - Balanced performance
- `mistral-small-latest` - Fast, efficient model
- `codestral-latest` - Optimized for code

## Usage

### Basic Chat

```python
async with MistralClient() as client:
    response = await client.chat("Hello!")
    print(response.content)
```

### Streaming

```python
async with MistralClient() as client:
    async for chunk in client.stream("Tell me a story"):
        print(chunk.content, end="")
```

### With System Prompt

```python
from src import Conversation

conv = Conversation(system="You are a helpful assistant.")
conv.add_user("Hello!")

async with MistralClient() as client:
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