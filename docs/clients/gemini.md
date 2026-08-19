# Gemini Client

The Gemini client provides access to Google's Gemini models.

## Configuration

```python
from padwan_llm.gemini import GeminiClient

client = GeminiClient(
    api_key="...",           # or set GEMINI_API_KEY env var
    model="gemini-2.5-flash", # default model
)
```

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with GeminiClient() as client:
    response, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
    print(response["content"])
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

## Thinking models

Gemini's reasoning models (e.g. `gemini-2.5-flash`, `gemini-2.5-pro`) can emit internal thought tokens alongside the final answer. Configure them via two fields:

- `thinking_config: ThinkingConfig | None` (Gemini-specific) — merged into every `generationConfig`. Set `thinkingBudget` to allocate tokens for reasoning and `includeThoughts=True` to have the thought parts streamed back.
- `on_thought: Callable[[str], None] | None` (defined on `LLMClientBase`, so any provider can adopt it) — called with each thought text chunk as it arrives. Thought chunks are **not** yielded as part of the normal text stream; this callback is the only way to see them.

```python
from padwan_llm import GeminiClient

thoughts: list[str] = []

async with GeminiClient(
    model="gemini-2.5-flash",
    on_thought=thoughts.append,
    thinking_config={"thinkingBudget": 2048, "includeThoughts": True},
) as client:
    stream = client.stream_chat([
        {"role": "user", "content": "What is 7 * 8? Think step by step."}
    ])
    async for chunk in stream:
        print(chunk, end="")

print("\n\nReasoning:", "".join(thoughts))
```

Without `includeThoughts=True`, the model may still think internally (consuming the budget) but won't emit the thoughts — the callback will never fire. Without a non-zero `thinkingBudget`, the model won't think at all.

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

## Realtime (Live API)

`RealtimeClient` with a Gemini model opens a speech-to-speech [Live API](https://ai.google.dev/api/live) session over a WebSocket. Audio is mono little-endian PCM16 — 16 kHz in (`LIVE_INPUT_SAMPLE_RATE`), 24 kHz out (`LIVE_OUTPUT_SAMPLE_RATE`). Requires the `realtime` extra.

```python
from padwan_llm import RealtimeClient

async with RealtimeClient(
    "gemini-3.1-flash-live-preview",
    instructions="Answer briefly.",
    voice="Puck",
) as conn:
    await conn.append_audio(pcm16_chunk)  # mono PCM16 @ 16 kHz
    async for message in conn:
        if audio := conn.audio_delta_bytes(message):
            playback.write(audio)  # PCM16 @ 24 kHz
        if conn.is_turn_complete(message):
            break
```

Automatic activity detection (server VAD) is the default. Pass `turn_detection=NO_TURN_DETECTION` to disable it and mark turns yourself with `conn.activity_start()` / `conn.activity_end()`; a mapping tunes the `automaticActivityDetection` fields (e.g. `{"silenceDurationMs": 500}`). Input/output transcription events arrive via `conn.input_transcript(message)` / `conn.output_transcript(message)`; disable them by constructing `GeminiRealtimeClient(transcription=False)` directly.
