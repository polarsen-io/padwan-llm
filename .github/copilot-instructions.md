# Copilot Instructions

When working on a model-drift PR, read the PR description first,
it contains the full drift report with per-provider lists
of candidate additions and removals.

## Model Literal rules

Each provider has a curated `Literal` type in its client file:

| Type | File |
|---|---|
| `OpenAIModel` | `padwan_llm/openai/client.py` |
| `GeminiModel` | `padwan_llm/gemini/client.py` |
| `MistralModel` | `padwan_llm/mistral/client.py` |
| `MistralEmbeddingModel` | `padwan_llm/mistral/client.py` |
| `MistralAudioModel` | `padwan_llm/mistral/client.py` |
| `GrokModel` | `padwan_llm/grok/client.py` |

**Add** a model ID when it appears under "Available but not tracked" AND:
- It is a public stable alias that belongs in the target `Literal`
- It is not a dated snapshot (no date suffix like `-0709`, `-20250514`)
- Its capability is supported by the target client

Do not remove model IDs listed under "Tracked but absent". Live provider responses
can vary by account, region, and permissions.

Do not remove deprecated models automatically. Leave deprecation handling for a
separate human-reviewed migration.

After editing a `Literal`, run `uv run pyright`, `uv run ruff check .`,
`uv run ruff format --check .`, and relevant tests.
