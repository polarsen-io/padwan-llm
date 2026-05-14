# Copilot Instructions

When working on a model-drift PR, read the PR description first — it contains the full drift report with per-provider lists of candidate additions and removals.

## Model Literal rules

Each provider has a curated `Literal` type in its client file:

| Type | File |
|---|---|
| `OpenAIModel` | `padwan_llm/openai/client.py` |
| `GeminiModel` | `padwan_llm/gemini/client.py` |
| `MistralModel` | `padwan_llm/mistral/client.py` |
| `GrokModel` | `padwan_llm/grok/client.py` |

**Add** a model ID when it appears under "Available but not tracked" AND:
- It is a stable or preview alias (e.g. `gemini-2.5-flash`, `grok-4-fast`)
- It is not a dated snapshot (no date suffix like `-0709`, `-20250514`)
- It is not an unversioned `*-latest` alias

**Remove** a model ID when it appears under "Tracked but absent" AND the drift report notes it is likely deprecated (not just region/account-gated).

After editing a `Literal`, run `uv run pyright` and `uv run ruff check .` to verify no type errors were introduced.
