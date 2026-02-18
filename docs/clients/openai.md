# OpenAI Client

The OpenAI client provides access to GPT models through the OpenAI API. It also serves as the base for other OpenAI-compatible providers (Grok, Mistral, etc.).

## Configuration

```python
from padwan_llm.openai import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    model="gpt-4o",    # default model
)
```

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with OpenAIClient() as client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
    print(text)
```

### Streaming

```python
from padwan_llm.conversation import Message

async with OpenAIClient() as client:
    stream = client.stream_chat([
        Message(role="user", content="Tell me a story")
    ])
    async for chunk in stream:
        print(chunk, end="")
```

### With System Prompt

```python
from padwan_llm import ConversationState

state = ConversationState(system="You are a helpful assistant.")
state.add_user_message("Hello!")

async with OpenAIClient() as client:
    text, usage = await client.complete_chat(state.messages)
    state.add_assistant_message(text)
    state.accumulate_usage(usage)
```

## Method Outputs

```python
text, usage = await client.complete_chat(messages)

stream = client.stream_chat(messages)
async for chunk in stream:
    ...
usage = stream.usage
```

## Batch Processing

OpenAI supports batch processing via a file-based API. Requests are serialized to JSONL, uploaded, and results are retrieved as a file once the batch completes.

### Creating a batch

```python
from padwan_llm.openai import OpenAIClient, BatchRequest

async with OpenAIClient() as client:
    requests = [
        BatchRequest(
            body={"messages": [{"role": "user", "content": "Question 1"}]},
            custom_id="q1",
        ),
        BatchRequest(
            body={"messages": [{"role": "user", "content": "Question 2"}]},
            custom_id="q2",
        ),
    ]
    job = await client.create_batch(requests, model="gpt-4o")
    print(job.id)  # e.g. "batch_abc123"
```

`BatchRequest` wraps a `CreateChatCompletionRequest` body and an optional `custom_id`. If `custom_id` is omitted, requests are auto-keyed as `request-0`, `request-1`, etc. The `model` field is injected into each JSONL line automatically.

### Polling for results

```python
from padwan_llm.openai import BatchResult

job = await client.get_batch(job.id)
if job.succeeded:
    results = await client.get_batch_results(job.output_file_id)
    for result in results:
        print(result.custom_id, result.content)
```

### Listing and cancelling

```python
jobs, next_cursor = await client.list_batches(limit=10)
job = await client.cancel_batch("batch_abc123")
```

### Batch types reference

| Type | Description |
|------|-------------|
| `BatchRequest` | Single request: `body`, `custom_id` |
| `BatchJob` | Job state: `id`, `status`, `input_file_id`, `output_file_id`, `request_counts`, `is_terminal`, `succeeded` |
| `BatchResult` | Parsed result: `custom_id`, `content`, `input_tokens`, `output_tokens`, `total_tokens` |
