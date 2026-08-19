# OpenAI Client

The OpenAI client provides access to GPT models through the OpenAI API. It also serves as the base for other OpenAI-compatible providers (Grok, Mistral, etc.).

## Configuration

```python
from padwan_llm.openai import OpenAIClient

client = OpenAIClient(
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    model="gpt-4o",  # default model
)
```

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with OpenAIClient() as client:
    response, usage = await client.complete_chat(
        [Message(role="user", content="Hello!")]
    )
    print(response["content"])
```

### Streaming

```python
from padwan_llm.conversation import Message

async with OpenAIClient() as client:
    stream = client.stream_chat([Message(role="user", content="Tell me a story")])
    async for chunk in stream:
        print(chunk, end="")
```

### With System Prompt

```python
from padwan_llm import ConversationState

state = ConversationState(system="You are a helpful assistant.")
state.add_user_message("Hello!")

async with OpenAIClient() as client:
    response, usage = await client.complete_chat(state.messages)
    state.add_assistant_message(response["content"])
    state.accumulate_usage(usage)
```

## Method Outputs

```python
response, usage = await client.complete_chat(messages)

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

## Realtime (Speech-to-Speech)

`RealtimeClient` opens speech-to-speech sessions with the GA `gpt-realtime` model over a WebSocket. Wire shapes follow the [OpenAI Realtime guide](https://platform.openai.com/docs/guides/realtime). Requires the `realtime` extra (niquests WebSocket support):

```bash
uv add "padwan-llm[realtime]"
```

Audio in both directions is mono little-endian PCM16 at 24 kHz (`REALTIME_SAMPLE_RATE`), the `gpt-realtime` native rate.

```python
from padwan_llm import RealtimeClient

client = RealtimeClient(
    api_key="sk-...",  # or set OPENAI_API_KEY
    model="gpt-realtime",  # default
    timeout=30.0,  # bounds the upgrade handshake only
)
```

`timeout` bounds only the WebSocket upgrade handshake. Reads on the open socket are unbounded, so the model can stay silent between turns without the connection being torn down.

`async with client as conn:` opens the underlying `niquests.AsyncSession`, performs the handshake, and yields the live `RealtimeConnection`; everything is closed on exit.

### Conversation with server VAD (default)

By default the server decides when you have stopped talking. Stream microphone audio in with `append_audio` and consume events by async-iterating the connection:

```python
from padwan_llm import RealtimeClient
from padwan_llm.openai import RealtimeServerEvent

async with RealtimeClient(instructions="Answer briefly.", voice="marin") as conn:
    await conn.append_audio(pcm16_chunk)  # mono PCM16 @ 24 kHz
    async for event in conn:
        if audio := conn.audio_delta_bytes(event):
            playback.write(audio)  # PCM16 bytes
        elif event["type"] == RealtimeServerEvent.RESPONSE_DONE:
            break
```

### Push-to-talk (manual turns)

Pass `NO_TURN_DETECTION` to disable server VAD, then drive each turn yourself:

```python
from padwan_llm import RealtimeClient
from padwan_llm.openai import NO_TURN_DETECTION

async with RealtimeClient(turn_detection=NO_TURN_DETECTION) as conn:
    await conn.append_audio(recorded_pcm16)
    await conn.commit_audio()
    await conn.create_response()
    async for event in conn:
        ...
```

### Client parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"gpt-realtime"` | Realtime model id |
| `instructions` | `None` | System prompt steering the voice agent |
| `voice` | `"marin"` | Spoken voice (`RealtimeVoice`) |
| `turn_detection` | `None` | `None`/falsy uses `server_vad`; a mapping (e.g. `{"type": "semantic_vad"}`) is sent verbatim; `NO_TURN_DETECTION` disables VAD |
| `transcription_model` | `"whisper-1"` | Transcribes your own speech; `None` disables |
| `output_modalities` | `("audio",)` | Response modalities; `("audio", "text")` also emits text |
| `sample_rate` | `24_000` | PCM16 sample rate in both directions |
| `timeout` | `30.0` | Bounds the upgrade handshake |
| `api_key` | `None` | Falls back to `OPENAI_API_KEY` |
| `base_url` | realtime endpoint | Custom `wss://` endpoint |

Constructing `OpenAIRealtimeClient` directly additionally accepts `session_kwargs=` to forward constructor arguments (e.g. proxies) to the managed `AsyncSession`. To reconfigure a live session, call `conn.configure(...)`.

### Server events

Iterating the connection yields each server event as a decoded JSON dict. `RealtimeServerEvent` names the common ones; any other event type passes through as a plain dict.

| Event | Meaning |
|-------|---------|
| `SESSION_CREATED` / `SESSION_UPDATED` | Session lifecycle acknowledgements |
| `SPEECH_STARTED` / `SPEECH_STOPPED` | Server VAD detected speech boundaries |
| `AUDIO_DELTA` / `AUDIO_DONE` | Model audio chunks (decode with `audio_delta_bytes`) and end of audio |
| `AUDIO_TRANSCRIPT_DELTA` / `AUDIO_TRANSCRIPT_DONE` | Transcript of the model's speech |
| `INPUT_TRANSCRIPT_DELTA` / `INPUT_TRANSCRIPT_COMPLETED` | Transcript of your speech |
| `RESPONSE_CREATED` / `RESPONSE_DONE` | Response lifecycle |
| `ERROR` | Server-reported error |

### Realtime limitations

- Reconnection is the caller's responsibility: a dropped socket ends iteration, and a new `connect()` starts a fresh session with no server-side memory of the previous one.
