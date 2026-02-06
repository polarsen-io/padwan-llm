# Gemini Client

The Gemini client provides access to Google's Gemini models.

## Configuration

```python
from src.gemini import GeminiClient

client = GeminiClient(
    api_key="...",           # or set GEMINI_API_KEY env var
    model="gemini-2.0-flash", # default model
)
```

## Available Models

- `gemini-2.0-flash` - Fast, efficient model
- `gemini-2.0-flash-thinking` - Enhanced reasoning
- `gemini-1.5-pro` - Previous generation pro model
- `gemini-1.5-flash` - Previous generation flash model

## Usage

### Basic Chat

```python
async with GeminiClient() as client:
    response = await client.chat("Hello!")
    print(response.content)
```

### Streaming

```python
async with GeminiClient() as client:
    async for chunk in client.stream("Tell me a story"):
        print(chunk.content, end="")
```

## Batch Processing

Gemini supports batch processing for large-scale requests:

```python
from src.gemini import GeminiBatch

batch = GeminiBatch(api_key="...")
results = await batch.process([
    "Question 1",
    "Question 2",
    "Question 3",
])
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