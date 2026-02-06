# CLI

Polarsen LLM includes a command-line interface with TUI mode for interactive conversations.

![Chat Demo](static/chat-demo.gif)

## Installation

Install with CLI support:

```bash
pip install polarsen-llm[cli]
```

## Usage

Start the TUI:

```bash
polarsen-llm
```

Or run a single command:

```bash
polarsen-llm chat:send "Hello, world!"
```

## Commands

### chat:send

Send a message and start a conversation:

```bash
polarsen-llm chat:send "What is Python?" -m gpt-4o
```

In TUI mode, this enters chat mode where you can continue the conversation. Press `Ctrl+C` to exit.

### chat:clear

Clear conversation history:

```bash
polarsen-llm chat:clear           # Clear all sessions
polarsen-llm chat:clear -m gpt-4o # Clear specific model session
```

### models

List available models:

```bash
polarsen-llm models              # List all models
polarsen-llm models -p openai    # Filter by provider
```

### info

Show library information:

```bash
polarsen-llm info
```

## Options

| Option | Description |
|--------|-------------|
| `-m`, `--model` | Model to use (default: gpt-4o-mini) |
| `-p`, `--provider` | Filter by provider (for `models` command) |