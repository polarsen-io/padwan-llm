# Observability (OpenTelemetry)

Padwan LLM ships opt-in OpenTelemetry instrumentation following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). It only depends on `opentelemetry-api`, behind the `otel` extra:

```bash
pip install "padwan-llm[otel]"
```

## Quick start

```python
from padwan_llm import otel

otel.instrument()  # uses the global tracer/meter providers
```

`instrument()` wraps `complete_chat` and `stream_chat` on every provider client (OpenAI, Gemini, Mistral, Grok, Anthropic). It is idempotent; call `otel.uninstrument()` to restore the original methods. Pass explicit providers to avoid globals:

```python
otel.instrument(tracer_provider=my_tracer_provider, meter_provider=my_meter_provider)
```

## Spans

Each chat call emits one `CLIENT` span named `chat <model>` (or `chat` when no model is set). For streams, the span starts on first iteration and ends when the stream completes, errors, is cancelled, or is abandoned.

| Attribute | Example | Notes |
|-----------|---------|-------|
| `gen_ai.operation.name` | `chat` | |
| `gen_ai.provider.name` | `openai`, `gcp.gemini`, `mistral_ai`, `x_ai`, `anthropic` | semconv well-known values |
| `gen_ai.request.model` | `gpt-4o` | omitted when no model is set |
| `gen_ai.request.temperature` | `0.7` | only when actually sent on the wire |
| `server.address` | `api.openai.com` | |
| `gen_ai.usage.input_tokens` | `10` | |
| `gen_ai.usage.output_tokens` | `20` | |
| `gen_ai.response.finish_reasons` | `["stop"]` | |
| `error.type` | `LLMError`, `CancelledError` | on failure, with `ERROR` status and a recorded exception |

## Metrics

| Instrument | Type | Unit | Attributes |
|------------|------|------|------------|
| `gen_ai.client.operation.duration` | Histogram | `s` | request attributes, plus `error.type` on failure |
| `gen_ai.client.token.usage` | Histogram | `{token}` | request attributes, plus `gen_ai.token.type` (`input` / `output`) |

## Not yet captured

- **Thoughts / reasoning**: no separate reasoning-token count or thinking-phase timing. Reasoning time is included in the overall span duration. Token accounting follows each provider's usage report: OpenAI-style APIs fold reasoning tokens into output tokens; Gemini reports thought tokens outside `candidatesTokenCount`, so they are absent from `gen_ai.usage.output_tokens`. The `on_thought` callback is not instrumented.
- **Tool usage**: `tool_calls` returned by a chat turn are not recorded on the span, and agent-level tool execution (`AgentSession`) emits no `execute_tool` spans.
- **Realtime, batch, embeddings**: only `complete_chat` / `stream_chat` are instrumented.
