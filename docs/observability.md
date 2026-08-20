# Observability (OpenTelemetry)

Padwan LLM ships opt-in OpenTelemetry instrumentation following the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai). It only depends on `opentelemetry-api`, behind the `otel` extra:

```bash
pip install "padwan-llm[otel]"
```

## Quick start

```python
from padwan_llm import otel

otel.instrument()  # uses the global tracer/meter providers
```

`instrument()` wraps every provider client (OpenAI, Gemini, Mistral, Grok, Anthropic): chat completions and streams, batch operations, embeddings, realtime sessions, plus agent tool execution. It is idempotent; call `otel.uninstrument()` to restore the original methods. Pass explicit providers to avoid globals:

```python
otel.instrument(tracer_provider=my_tracer_provider, meter_provider=my_meter_provider)
```

## Spans

Each chat call emits one `CLIENT` span named `chat <model>` (or `chat` when no model is set). For streams, the span starts on first iteration and ends when the stream completes, errors, is cancelled, or is abandoned.

| Attribute | Example | Notes |
|-----------|---------|-------|
| `gen_ai.operation.name` | `chat` | |
| `gen_ai.provider.name` | `openai`, `gcp.gemini`, `mistral_ai`, `x_ai`, `anthropic` | semconv well-known values; OpenAI-compatible endpoints report `openai`, distinguished by `server.address` |
| `gen_ai.request.model` | `gpt-4o` | omitted when no model is set |
| `gen_ai.request.temperature` | `0.7` | only when actually sent on the wire |
| `server.address` | `api.openai.com` | |
| `gen_ai.usage.input_tokens` | `10` | |
| `gen_ai.usage.output_tokens` | `20` | |
| `gen_ai.usage.cache_read.input_tokens` | `3` | when the provider reports cached prompt tokens |
| `gen_ai.usage.reasoning.output_tokens` | `5` | when the provider reports thought/reasoning tokens separately¹ |
| `gen_ai.response.finish_reasons` | `["stop"]` | |
| `padwan_llm.response.tool_names` | `["get_weather"]` | tool calls requested by the model (custom attribute) |
| `padwan_llm.thinking.duration` | `1.2` | seconds between the first and last `on_thought` chunk of a stream (custom attribute) |
| `error.type` | `LLMError`, `CancelledError` | on failure, with `ERROR` status and a recorded exception |

¹ Token accounting follows each provider's usage report: OpenAI-style APIs count reasoning tokens inside `output_tokens`; Gemini reports thought tokens outside `candidatesTokenCount`, so they are not part of `gen_ai.usage.output_tokens`. Anthropic does not report a separate count. Per the Anthropic-specific conventions, `gen_ai.usage.input_tokens` includes cache read/write tokens (which Anthropic's raw `input_tokens` excludes).

## Tool execution

Each tool dispatched by an `AgentSession` emits an `execute_tool <name>` span following the semconv `execute_tool` operation:

| Attribute | Example |
|-----------|---------|
| `gen_ai.operation.name` | `execute_tool` |
| `gen_ai.tool.name` | `get_weather` |
| `gen_ai.tool.type` | `function` |
| `gen_ai.tool.call.id` | `call_1` |
| `gen_ai.tool.description` | `Get the current weather` |

## Embeddings, batch, and realtime

- **Embeddings**: `MistralClient.fetch_embeddings` emits an `embeddings <model>` span (`gen_ai.operation.name=embeddings`).
- **Batch**: batch operations (`create_batch`, `get_batch`, `list_batches`, `cancel_batch`, and the OpenAI/Grok file helpers) emit a span named after the operation. No model attribute is set — batch requests carry their own per-request models.
- **Realtime**: a `realtime <model>` span covers the whole `RealtimeClient` session, from connect to close, with connect failures recorded as errors.

All of these record the `gen_ai.client.operation.duration` histogram with their `gen_ai.operation.name`.

## Metrics

| Instrument | Type | Unit | Attributes |
|------------|------|------|------------|
| `gen_ai.client.operation.duration` | Histogram | `s` | request attributes, plus `error.type` on failure |
| `gen_ai.client.token.usage` | Histogram | `{token}` | request attributes, plus `gen_ai.token.type` (`input` / `output`) |

## Semconv coverage

Support status for each section of the [GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/README.md):

| Convention | Status | Notes |
|------------|--------|-------|
| [Model spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md) | ✅ Supported | `chat` and `embeddings` spans with semconv naming, request/usage attributes (incl. `reasoning.output_tokens`, `cache_read.input_tokens`). Batch and realtime spans use custom operation names — semconv defines none for them. |
| [Agent spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md) | 🟡 Partial | `execute_tool` spans for `AgentSession` tool dispatch. No `invoke_agent` / `create_agent` spans — `AgentSession` turns are not wrapped. |
| [Metrics](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md) | 🟡 Partial | `gen_ai.client.operation.duration` and `gen_ai.client.token.usage`. No `time_to_first_chunk` / `time_per_output_chunk`; server metrics don't apply to a client library. |
| [Events](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md) | ❌ Not supported | No `gen_ai.client.inference.operation.details` event: message/prompt content (`gen_ai.input.messages`, `gen_ai.output.messages`) is deliberately never captured — it is opt-in in the spec and sensitive by nature. |
| [Exceptions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-exceptions.md) | 🟡 Partial | Failures set `error.type`, `ERROR` status, and a recorded span exception. The dedicated `gen_ai.client.operation.exception` log event is not emitted. |
| [Anthropic](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/anthropic.md) | ✅ Supported | Cache-inclusive `input_tokens` accounting per the provider conventions. No `gen_ai.usage.cache_write.input_tokens` breakdown or `gen_ai.request.reasoning.level`. |
| [OpenAI](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/openai.md) | 🟡 Partial | Core conventions apply; vendor extras (`openai.request.service_tier`, `openai.response.system_fingerprint`, `openai.api.type`) are not emitted. |
| [Azure AI Inference](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/azure-ai-inference.md) | — N/A | Not a padwan-llm provider. |
| [AWS Bedrock](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/aws-bedrock.md) | — N/A | Not a padwan-llm provider. |
| [MCP](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/mcp.md) | ❌ Not supported | `McpStreamable` / `McpStdio` transports emit no `mcp.*` spans or metrics. |
