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

## Method Outputs

```python
text, usage = await client.complete_chat(messages)

stream = client.stream_chat(messages)
async for chunk in stream:
    ...
usage = stream.usage
```
