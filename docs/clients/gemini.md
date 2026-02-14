# Gemini Client

The Gemini client provides access to Google's Gemini models.

## Configuration

```python
from padwan_llm.gemini import GeminiClient

client = GeminiClient(
    api_key="...",           # or set GEMINI_API_KEY env var
    model="gemini-2.0-flash", # default model
)
```

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with GeminiClient() as client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
    print(text)
```

### Streaming

```python
from padwan_llm.conversation import Message

async with GeminiClient() as client:
    stream = client.stream_chat([
        Message(role="user", content="Tell me a story")
    ])
    async for chunk in stream:
        print(chunk, end="")
```

## Batch Processing

Gemini supports batch processing for large-scale requests via methods on `GeminiClient`.

### Creating a batch

```python
from padwan_llm.gemini import GeminiClient, BatchRequest

async with GeminiClient() as client:
    requests = [
        BatchRequest(
            contents=[{"role": "user", "parts": [{"text": "Question 1"}]}],
            key="q1",
        ),
        BatchRequest(
            contents=[{"role": "user", "parts": [{"text": "Question 2"}]}],
            key="q2",
        ),
    ]
    job = await client.create_batch(requests, display_name="my-batch")
    print(job.name)  # e.g. "batches/123456"
```

`BatchRequest` accepts optional `generation_config` and `system_instruction` fields. If `key` is omitted, requests are auto-keyed as `request-0`, `request-1`, etc.

### Polling for results

```python
from padwan_llm.gemini import BatchResult

job = await client.get_batch(job.name)
if job.succeeded:
    for resp in job.inlined_responses or []:
        result = BatchResult.from_inlined_response(resp)
        print(result.key, result.content)
```

### Listing and cancelling

```python
jobs, next_token = await client.list_batches(page_size=10)
await client.cancel_batch("batches/123456")
```

### Batch types reference

| Type | Description |
|------|-------------|
| `BatchRequest` | Single request: `contents`, `generation_config`, `system_instruction`, `key` |
| `BatchJob` | Job state: `name`, `state`, `dest`, `stats`, `is_terminal`, `succeeded` |
| `BatchResult` | Parsed result: `key`, `content`, `input_tokens`, `output_tokens`, `total_tokens` |
