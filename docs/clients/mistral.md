# Mistral Client

The Mistral client provides access to Mistral AI models. It inherits from [`OpenAIClient`](openai.md) since Mistral uses an OpenAI-compatible API.

## Configuration

```python
from padwan_llm.mistral import MistralClient

client = MistralClient(
    api_key="...",           # or set MISTRAL_API_KEY env var
    model="mistral-large-latest",  # default model
)
```

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with MistralClient() as client:
    text, usage = await client.complete_chat([
        Message(role="user", content="Hello!")
    ])
    print(text)
```

### Streaming

```python
from padwan_llm.conversation import Message

async with MistralClient() as client:
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

async with MistralClient() as client:
    text, usage = await client.complete_chat(state.messages)
    state.add_assistant_message(text)
    state.accumulate_usage(usage)
```

## Audio Transcription

Transcribe audio using the `voxtral-mini-latest` model.

```python
async with MistralClient() as client:
    # From a local file
    result = await client.transcribe(file="recording.mp3")
    print(result["text"])

    # From a URL
    result = await client.transcribe(file_url="https://example.com/audio.mp3")

    # From an uploaded file ID
    result = await client.transcribe(file_id="file-abc123")
```

Exactly one of `file`, `file_id`, or `file_url` must be provided. `file` accepts a path (`str`/`Path`) or raw `bytes`.

Optional parameters: `language`, `temperature`, `diarize` (speaker detection), and `timestamp_granularities` (`["segment"]` and/or `["word"]`).

```python
result = await client.transcribe(
    file="meeting.mp3",
    language="en",
    diarize=True,
    timestamp_granularities=["segment", "word"],
)
for segment in result.get("segments", []):
    print(f"[{segment['start']:.1f}s] {segment['text']}")
```

## Embeddings

Generate text embeddings using the `mistral-embed` model.

```python
async with MistralClient() as client:
    resp = await client.fetch_embeddings("Hello, world!")
    # Or batch multiple texts
    resp = await client.fetch_embeddings(["text 1", "text 2"])
    # resp is an EmbeddingResponse; extract vectors from resp["data"]
    vectors = [item["embedding"] for item in resp["data"]]
```
