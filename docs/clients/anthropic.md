# Anthropic Client

The Anthropic client provides access to Claude models through the native
[Messages API](https://platform.claude.com/docs/en/api/messages). It is a
standalone client (not OpenAI-compatible).

## Configuration

```python
from padwan_llm.anthropic import AnthropicClient

client = AnthropicClient(
    api_key="...",  # or set ANTHROPIC_API_KEY env var
    model="claude-opus-4-8",  # default model
    max_tokens=4096,  # required by the Messages API, per-response cap
)
```

!!! note "Sampling parameters"
    Current Claude models reject non-default sampling parameters, so the
    inherited `temperature` field is never sent — steer behavior via the
    prompt instead.

## Usage

### Basic Chat

```python
from padwan_llm.conversation import Message

async with AnthropicClient() as client:
    response, usage = await client.complete_chat(
        [Message(role="user", content="Hello!")]
    )
    print(response["content"])
```

### Streaming

```python
from padwan_llm.conversation import Message

async with AnthropicClient() as client:
    stream = client.stream_chat([Message(role="user", content="Tell me a story")])
    async for chunk in stream:
        print(chunk, end="")
```

### With System Prompt

System messages are translated to the Messages API's top-level `system` field.

```python
from padwan_llm import ConversationState

state = ConversationState(system="You are a helpful assistant.")
state.add_user_message("Hello!")

async with AnthropicClient() as client:
    response, usage = await client.complete_chat(state.messages)
    state.add_assistant_message(response["content"])
    state.accumulate_usage(usage)
```

### Tool Calling

`tool_use` blocks map to the shared `ToolCall` shape; send results back as
`ToolResultMessage`s.

```python
from padwan_llm.models import ToolDefinition

WEATHER_TOOL: ToolDefinition = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

async with AnthropicClient() as client:
    response, _ = await client.complete_chat(
        [{"role": "user", "content": "Weather in Paris?"}],
        tools=[WEATHER_TOOL],
    )
    call = response["tool_calls"][0]

    response2, _ = await client.complete_chat(
        [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": response["content"], "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": "get_weather",
                "content": "18°C",
            },
        ],
        tools=[WEATHER_TOOL],
    )
```

### Thinking

Claude models think adaptively by default. Summarized thinking (when exposed
by the model) is forwarded to the `on_thought` callback and never leaks into
the answer text.

```python
async with AnthropicClient(
    model="claude-opus-4-8",
    on_thought=lambda t: print(f"[thinking] {t}"),
) as client:
    response, _ = await client.complete_chat(
        [{"role": "user", "content": "What is 7 * 8?"}]
    )
```

## Method Outputs

```python
response, usage = await client.complete_chat(messages)
# usage["cached"] carries cache_read_input_tokens when present

stream = client.stream_chat(messages)
async for chunk in stream:
    ...
usage = stream.usage
tool_calls = stream.tool_calls
```
