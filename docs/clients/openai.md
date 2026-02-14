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
