from unittest.mock import MagicMock

import pytest
from langfuse.types import (
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanData,
    OtelSpanIdentifier,
    OtelSpanPatch,
)
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.util.types import AttributeValue

import padwan_llm.langfuse as langfuse_integration
from padwan_llm import otel
from padwan_llm._json import dumps as _json_dumps


def _span_params(
    attributes: dict[str, AttributeValue], *, scope: str = "padwan_llm"
) -> tuple[OtelSpanIdentifier, MaskOtelSpansParams]:
    identifier = OtelSpanIdentifier(trace_id="0" * 32, span_id="1" * 16)
    span = OtelSpanData(
        trace_id=identifier.trace_id,
        span_id=identifier.span_id,
        parent_span_id=None,
        name="test",
        instrumentation_scope_name=scope,
        instrumentation_scope_version="1",
        attributes=attributes,
        resource_attributes={},
    )
    return identifier, MaskOtelSpansParams(spans={identifier: span})


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        pytest.param("chat", "generation", id="chat"),
        pytest.param("embeddings", "embedding", id="embeddings"),
        pytest.param("invoke_agent", "agent", id="agent"),
        pytest.param("execute_tool", "tool", id="tool"),
        pytest.param("realtime", "span", id="fallback_span"),
    ],
)
def test_span_adapter_maps_observation_type(operation: str, expected: str):
    identifier, params = _span_params({"gen_ai.operation.name": operation})

    result = langfuse_integration._SpanAdapter()(params=params)

    assert result is not None
    patch = result.span_patches[identifier]
    assert patch is not None
    assert patch.set_attributes["langfuse.observation.type"] == expected


_MESSAGES = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
_SYSTEM = [{"type": "text", "content": "be brief"}]
_TOOLS = [{"type": "function", "name": "search"}]


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        pytest.param(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.input.messages": _json_dumps(_MESSAGES),
                "gen_ai.system_instructions": _json_dumps(_SYSTEM),
                "gen_ai.tool.definitions": _json_dumps(_TOOLS),
                "gen_ai.output.messages": _json_dumps([{"role": "assistant"}]),
                "gen_ai.request.model": "gpt-test",
                "gen_ai.conversation.id": "session-1",
            },
            {
                "langfuse.observation.type": "generation",
                "langfuse.observation.input": _json_dumps(
                    {
                        "messages": _MESSAGES,
                        "system_instructions": _SYSTEM,
                        "tools": _TOOLS,
                    }
                ),
                "langfuse.observation.output": _json_dumps([{"role": "assistant"}]),
                "langfuse.session.id": "session-1",
            },
            id="chat_sections_and_session",
        ),
        pytest.param(
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.call.arguments": '{"query":"test"}',
                "gen_ai.tool.call.result": '{"result":"ok"}',
            },
            {
                "langfuse.observation.type": "tool",
                "langfuse.observation.input": '{"query":"test"}',
                "langfuse.observation.output": '{"result":"ok"}',
            },
            id="tool_call",
        ),
    ],
)
def test_span_adapter_maps_content(
    attributes: dict[str, AttributeValue], expected: dict[str, AttributeValue]
):
    identifier, params = _span_params(attributes)

    result = langfuse_integration._SpanAdapter()(params=params)

    assert result is not None
    patch = result.span_patches[identifier]
    assert patch is not None
    assert patch.set_attributes == expected


def test_span_adapter_applies_mask_before_mapping():
    identifier, params = _span_params(
        {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": '[{"content":"secret"}]',
        }
    )

    def mask(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
        return MaskOtelSpansResult(
            span_patches={
                next(iter(params.spans)): OtelSpanPatch(
                    delete_attributes=("gen_ai.input.messages",)
                )
            }
        )

    result = langfuse_integration._SpanAdapter(mask)(params=params)

    assert result is not None
    patch = result.span_patches[identifier]
    assert patch is not None
    assert "langfuse.observation.input" not in patch.set_attributes
    assert patch.delete_attributes == ("gen_ai.input.messages",)


def test_span_adapter_ignores_other_instrumentation_scopes():
    _, params = _span_params(
        {"gen_ai.operation.name": "chat"}, scope="other.instrumentation"
    )

    assert langfuse_integration._SpanAdapter()(params=params) is None


@pytest.mark.parametrize(
    ("user_filter", "expected"),
    [
        pytest.param(None, True, id="padwan_mcp_span_included"),
        pytest.param(lambda _span: False, False, id="user_filter_denies"),
    ],
)
def test_span_filter(user_filter, expected: bool):
    span = MagicMock(spec=ReadableSpan)
    span.instrumentation_scope = InstrumentationScope("padwan_llm")
    span.attributes = {"mcp.method.name": "initialize"}

    assert langfuse_integration._SpanFilter(user_filter)(span) is expected


def test_instrument_owns_lifecycle(monkeypatch):
    client = MagicMock()
    langfuse = MagicMock(return_value=client)
    instrument_otel = MagicMock()
    uninstrument_otel = MagicMock()
    monkeypatch.setattr(langfuse_integration, "Langfuse", langfuse)
    monkeypatch.setattr(otel, "is_instrumented", lambda: False)
    monkeypatch.setattr(otel, "instrument", instrument_otel)
    monkeypatch.setattr(otel, "uninstrument", uninstrument_otel)
    provider = TracerProvider()

    integration = langfuse_integration.instrument(
        public_key="<PUBLIC_KEY>",
        secret_key="<SECRET_KEY>",
        tracer_provider=provider,
        capture_content=True,
    )
    integration.flush()
    integration.shutdown()
    integration.shutdown()

    assert integration.client is client
    instrument_otel.assert_called_once_with(
        tracer_provider=provider, capture_content=True
    )
    client.flush.assert_called_once_with()
    uninstrument_otel.assert_called_once_with()
    client.shutdown.assert_called_once_with()
    options = langfuse.call_args.kwargs
    assert isinstance(options["mask_otel_spans"], langfuse_integration._SpanAdapter)
    assert isinstance(options["should_export_span"], langfuse_integration._SpanFilter)


def test_instrument_rejects_active_otel(monkeypatch):
    monkeypatch.setattr(otel, "is_instrumented", lambda: True)

    with pytest.raises(RuntimeError, match="already active"):
        langfuse_integration.instrument()
