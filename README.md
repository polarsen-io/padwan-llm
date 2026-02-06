# Polarsen LLM

Unified async client for OpenAI, Gemini, Mistral, and Grok APIs.

## Installation

```bash
# Library only
pip install polarsen-llm

# With CLI
pip install polarsen-llm[cli]
# or
uvx polarsen-llm[cli]
```

## Library Usage

```python
from src import LLMClient, ConversationState

async with LLMClient(model="gpt-4o") as client:
    response, usage = await client.complete({
        "messages": [{"role": "user", "content": "Hello!"}]
    })
```

### Streaming with ConversationState

```python
from src import LLMClient, ConversationState

state = ConversationState(system="You are helpful.")

async with LLMClient(model="gpt-4o") as client:
    state.add_user_message("Hello!")

    chat_stream = client.stream_chat(state.messages)
    async for text in chat_stream:
        print(text, end="", flush=True)

    if chat_stream.usage:
        state.accumulate_usage(chat_stream.usage)
```

## CLI Usage

```bash
# List available models
polarsen-llm models
polarsen-llm models -p gemini

# Show library info
polarsen-llm info

# Chat with an LLM
polarsen-llm chat send "Hello!" -m gpt-4o-mini
```

### TUI Mode

![Chat Demo](docs/static/chat-demo.gif)

```bash
polarsen-llm
```

- Type `/` to see available commands
- Arrow keys to navigate, Tab to confirm
- Ctrl+C twice to exit

### Commands

| Command | Description |
|---------|-------------|
| `models [-p provider]` | List available models |
| `info` | Show library info |
| `chat send <msg> [-m model]` | Send a message |
| `chat clear [-m model]` | Clear history |
| `chat history [-m model]` | Show history |
| `batch create -p <prompt>` | Create Gemini batch job |
| `batch status -j <job>` | Check batch status |
| `batch poll -j <job>` | Poll until completion |

## Supported Models

| Provider | Models |
|----------|--------|
| OpenAI | gpt-4o, gpt-4o-mini, o1, o3-mini, ... |
| Gemini | gemini-2.0-flash, gemini-1.5-pro, ... |
| Mistral | mistral-large-latest, codestral-latest, ... |
| Grok | grok-3, grok-3-mini, ... |

## Environment Variables

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROK_API_KEY=...
```
