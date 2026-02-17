# Padwan LLM

Lightweight, unified async client for OpenAI, Gemini, Mistral, Grok, and any OpenAI-compatible API.
Single dependency ([niquests](https://github.com/jawah/niquests)), automatic HTTP/2 and HTTP/3 negotiation.

For the full interactive CLI/TUI, use the separate [`padwan-cli`](https://github.com/polarsen-io/padwan-cli) package.

<img alt="Chat demo" src="https://github.com/polarsen-io/padwan-cli/raw/master/docs/static/chat.gif" width="500"/>

## Installation

```bash
pip install padwan-llm
```

## Library Usage

```python
from padwan_llm import LLMClient, ConversationState

async with LLMClient(model="gpt-4o") as client:
    response, usage = await client.complete({
        "messages": [{"role": "user", "content": "Hello!"}]
    })
```

### Streaming with ConversationState

```python
from padwan_llm import LLMClient, ConversationState

state = ConversationState(system="You are a concise assistant, helping me in my daily tasks.")

async with LLMClient(model="gpt-4o") as client:
    state.add_user_message("What's python?")

    chat_stream = client.stream_chat(state.messages)
    async for text in chat_stream:
        print(text, end="", flush=True)

    if chat_stream.usage:
        state.accumulate_usage(chat_stream.usage)
```

## One-Shot Command

```bash
export OPENAI_API_KEY=...

padwan-llm "Hello!" -m gpt-4o-mini

# Or without installing:
uvx padwan-llm "Hello!" -m gpt-4o-mini
```

## Supported Models

Auto-detected providers: **OpenAI**, **Gemini**, **Mistral**, **Grok**.

Any OpenAI-compatible API (Groq, Together AI, Ollama, vLLM, ...) is supported via `OpenAIClient`.

## Environment Variables

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
MISTRAL_API_KEY=...
GROK_API_KEY=...
```
