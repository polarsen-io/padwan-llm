# OpenAI Realtime Client

Speech-to-speech sessions with the GA `gpt-realtime` model over a WebSocket. Wire shapes follow the [OpenAI Realtime guide](https://platform.openai.com/docs/guides/realtime).

Requires the `realtime` extra (niquests WebSocket support):

```bash
uv add "padwan-llm[realtime]"
```

## Configuration

```python
from padwan_llm.openai import RealtimeClient

client = RealtimeClient(
    api_key="sk-...",      # or set OPENAI_API_KEY
    model="gpt-realtime",  # default
    timeout=30.0,          # bounds the upgrade handshake only
)
```

`timeout` bounds only the WebSocket upgrade handshake. Reads on the open socket are unbounded, so the model can stay silent between turns without the connection being torn down.

## Usage

Audio in both directions is mono little-endian PCM16 at 24 kHz (`REALTIME_SAMPLE_RATE`), the `gpt-realtime` native rate.

### Conversation with server VAD (default)

By default the server decides when you have stopped talking. Stream microphone audio in with `append_audio` and consume events by async-iterating the connection:

```python
from padwan_llm.openai import RealtimeClient, RealtimeServerEvent

client = RealtimeClient()
async with client.connect(instructions="Answer briefly.", voice="marin") as conn:
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
from padwan_llm.openai import NO_TURN_DETECTION, RealtimeClient

client = RealtimeClient()
async with client.connect(turn_detection=NO_TURN_DETECTION) as conn:
    await conn.append_audio(recorded_pcm16)
    await conn.commit_audio()
    await conn.create_response()
    async for event in conn:
        ...
```

### Reusing an HTTP session

`connect()` creates and closes its own `niquests.AsyncSession` per connection. Pass `session=` to reuse a caller-owned session (left open on exit), or `session_kwargs=` to forward constructor arguments to the internally managed one; the two are mutually exclusive.

```python
import niquests

async with niquests.AsyncSession() as session:
    async with client.connect(session=session) as conn:
        ...
    # session is still open here for the next connect()
```

## `connect()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `instructions` | `None` | System prompt steering the voice agent |
| `voice` | `"marin"` | Spoken voice (`RealtimeVoice`) |
| `turn_detection` | `None` | `None`/falsy uses `server_vad`; a mapping (e.g. `{"type": "semantic_vad"}`) is sent verbatim; `NO_TURN_DETECTION` disables VAD |
| `transcription_model` | `"whisper-1"` | Transcribes your own speech; `None` disables |
| `output_modalities` | `("audio",)` | Response modalities; `("audio", "text")` also emits text |
| `sample_rate` | `24_000` | PCM16 sample rate in both directions |
| `session` | `None` | Caller-owned `AsyncSession`; left open on exit |
| `session_kwargs` | `None` | Constructor arguments for the internally managed session |

## Server events

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

## Limitations

- OpenAI only; there is no Gemini Live equivalent.
- Reconnection is the caller's responsibility: a dropped socket ends iteration, and a new `connect()` starts a fresh session with no server-side memory of the previous one.
