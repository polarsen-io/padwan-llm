# Grok Client

The Grok client provides access to xAI's Grok models. It inherits from [`OpenAIClient`](openai.md) since Grok uses an OpenAI-compatible API.

## Configuration

```python
from padwan_llm.grok import GrokClient

client = GrokClient(
    api_key="...",      # or set GROK_API_KEY env var
    model="grok-3",     # default model
)
```

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with GrokClient() as client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
    print(text)
```

### Streaming

```python
from padwan_llm.conversation import Message

async with GrokClient() as client:
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

async with GrokClient() as client:
    text, usage = await client.complete_chat(state.messages)
    state.add_assistant_message(text)
    state.accumulate_usage(usage)
```

## Batch API

Grok uses xAI's native batch API, which differs from OpenAI's file-upload approach.
Requests are submitted directly via JSON and results are fetched per-batch.

### Create a batch

```python
from padwan_llm.grok import GrokClient, GrokBatchRequest

requests = [
    GrokBatchRequest(
        body={"messages": [{"role": "user", "content": "Summarize X"}]},
        custom_id="req-1",
    ),
    GrokBatchRequest(
        body={"messages": [{"role": "user", "content": "Summarize Y"}]},
        custom_id="req-2",
    ),
]

async with GrokClient() as client:
    job = await client.create_batch(requests, name="summaries")
    print(job.batch_id, job.num_requests)
```

### Check status and fetch results

```python
async with GrokClient() as client:
    job = await client.get_batch("batch-abc")

    if job.succeeded:
        results, _ = await client.get_batch_results(job.batch_id)
        for r in results:
            print(r.custom_id, r.content)
```

### List and cancel batches

```python
async with GrokClient() as client:
    jobs, next_token = await client.list_batches(limit=10)
    job = await client.cancel_batch("batch-abc")
```

## Method Outputs

```python
text, usage = await client.complete_chat(messages)

stream = client.stream_chat(messages)
async for chunk in stream:
    ...
usage = stream.usage
```
