import asyncio
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    Histogram,
    HistogramDataPoint,
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from padwan_llm import McpStreamable, McpTool, otel
from padwan_llm._base import RealtimeClientBase
from padwan_llm._json import dumps as _json_dumps, loads as _json_loads
from padwan_llm.errors import LLMError, Provider
from padwan_llm.gemini import GeminiClient
from padwan_llm.mistral import MistralClient
from padwan_llm.openai import OpenAIClient
from tests.test_agent import FakeChatStream, make_session, make_tool_call

USAGE = {"total_tokens": 30, "prompt_tokens": 10, "completion_tokens": 20}


@pytest.fixture
def otel_setup():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    otel.instrument(tracer_provider=tracer_provider, meter_provider=meter_provider)
    yield exporter, reader
    otel.uninstrument()


@pytest.fixture
def client():
    c = OpenAIClient(model="gpt-4o", api_key="test")
    c._session = AsyncMock()
    return c


def _metric_names(reader: InMemoryMetricReader) -> set[str]:
    data = reader.get_metrics_data()
    if data is None:
        return set()
    return {
        metric.name
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
    }


def _histogram_points(
    reader: InMemoryMetricReader, name: str
) -> list[HistogramDataPoint]:
    data = reader.get_metrics_data()
    if data is None:
        return []
    points: list[HistogramDataPoint] = []
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name and isinstance(metric.data, Histogram):
                    points.extend(metric.data.data_points)
    return points


def _histogram_count(reader: InMemoryMetricReader, name: str) -> int:
    return sum(point.count for point in _histogram_points(reader, name))


def _histogram_sum(reader: InMemoryMetricReader, name: str) -> int | float:
    return sum(point.sum for point in _histogram_points(reader, name))


@pytest.mark.parametrize(
    ("payload_extra", "expected_attrs"),
    [
        pytest.param(
            {},
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "openai",
                "gen_ai.request.model": "gpt-4o",
                "server.address": "api.openai.com",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 20,
                "gen_ai.response.finish_reasons": ("stop",),
            },
            id="semconv_base",
        ),
        pytest.param(
            {
                "usage": {
                    **USAGE,
                    "completion_tokens_details": {"reasoning_tokens": 5},
                    "prompt_tokens_details": {"cached_tokens": 3},
                }
            },
            {
                "gen_ai.usage.reasoning.output_tokens": 5,
                "gen_ai.usage.cache_read.input_tokens": 3,
            },
            id="reasoning_and_cached_tokens",
        ),
        pytest.param(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "padwan_llm.response.tool_names": ("get_weather",),
                "gen_ai.response.finish_reasons": ("tool_calls",),
            },
            id="tool_names",
        ),
        pytest.param(
            {"service_tier": "default", "system_fingerprint": "fp_123"},
            {
                "openai.api.type": "chat_completions",
                "openai.response.service_tier": "default",
                "openai.response.system_fingerprint": "fp_123",
            },
            id="openai_vendor_extras",
        ),
    ],
)
async def test_complete_chat_span_attributes(
    otel_setup, client, make_resp, payload_extra: dict, expected_attrs: dict
):
    exporter, reader = otel_setup
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
        **payload_extra,
    }
    client._session.post.return_value = make_resp(200, payload)

    await client.complete_chat([{"role": "user", "content": "hey"}])

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    attrs = dict(span.attributes or {})
    assert expected_attrs.items() <= attrs.items()
    assert _metric_names(reader) == {
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    }


async def test_stream_chat_span(otel_setup, client, make_sse_event, make_sse_resp):
    exporter, reader = otel_setup
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " "}}]},
        {
            "choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}],
            "usage": USAGE,
            "service_tier": "flex",
            "system_fingerprint": "fp_stream",
        },
    ]
    events = [make_sse_event(_json_dumps(c)) for c in chunks]
    client._session.post.return_value = make_sse_resp(events)

    stream = client.stream_chat(
        [{"role": "user", "content": "hey"}], extra_params={"service_tier": "flex"}
    )
    assert exporter.get_finished_spans() == ()

    text = [t async for t in stream]

    assert "".join(text) == "Hello world"
    assert stream.usage == {"total": 30, "input": 10, "output": 20}
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert attrs["gen_ai.response.time_to_first_chunk"] > 0
    assert attrs["openai.api.type"] == "chat_completions"
    assert attrs["openai.request.service_tier"] == "flex"
    assert attrs["openai.response.service_tier"] == "flex"
    assert attrs["openai.response.system_fingerprint"] == "fp_stream"
    assert {
        "gen_ai.client.operation.time_to_first_chunk",
        "gen_ai.client.operation.time_per_output_chunk",
    } <= _metric_names(reader)
    assert (
        _histogram_count(reader, "gen_ai.client.operation.time_per_output_chunk") == 2
    )


@pytest.mark.parametrize(
    ("model", "side_effect", "raises", "expected_name"),
    [
        pytest.param(None, None, LLMError, "chat", id="missing_model"),
        pytest.param(
            "gpt-4o",
            asyncio.CancelledError(),
            asyncio.CancelledError,
            "chat gpt-4o",
            id="cancelled",
        ),
    ],
)
async def test_complete_chat_error_records_error_type(
    otel_setup,
    model: str | None,
    side_effect: BaseException | None,
    raises: type[BaseException],
    expected_name: str,
):
    exporter, _ = otel_setup
    c = OpenAIClient(model=model, api_key="test")
    c._session = AsyncMock()
    c._session.post.side_effect = side_effect

    with pytest.raises(raises):
        await c.complete_chat([{"role": "user", "content": "hey"}])

    (span,) = exporter.get_finished_spans()
    assert span.name == expected_name
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == raises.__name__


async def test_stream_error_records_error(otel_setup, client):
    exporter, _ = otel_setup
    resp = AsyncMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock(return_value=resp)
    resp.extension = None
    client._session.post.return_value = resp
    stream = client.stream_chat([{"role": "user", "content": "hey"}])

    with pytest.raises(LLMError):
        _ = [t async for t in stream]

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "LLMError"


async def test_stream_abandoned_closes_span_cleanly(
    otel_setup, client, make_sse_event, make_sse_resp
):
    exporter, _ = otel_setup
    events = [
        make_sse_event(_json_dumps({"choices": [{"delta": {"content": t}}]}))
        for t in ("Hello", " world")
    ]
    client._session.post.return_value = make_sse_resp(events)
    stream = client.stream_chat([{"role": "user", "content": "hey"}])

    it = stream.__aiter__()
    assert await anext(it) == "Hello"
    await it.aclose()

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert "error.type" not in dict(span.attributes or {})


async def test_stream_cancellation_records_error(otel_setup, client):
    exporter, _ = otel_setup

    async def fake_stream(_body):
        yield {"choices": [{"delta": {"content": "x"}}]}
        await asyncio.Event().wait()

    client.stream = fake_stream
    stream = client.stream_chat([{"role": "user", "content": "hey"}])

    async def consume():
        async for _ in stream:
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "CancelledError"


async def test_instrument_raises_when_already_active(otel_setup):
    with pytest.raises(RuntimeError, match="already active"):
        otel.instrument()


async def test_uninstrument_restores_methods(otel_setup, client, make_resp):
    exporter, _ = otel_setup
    otel.uninstrument()
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)

    await client.complete_chat([{"role": "user", "content": "hey"}])

    assert exporter.get_finished_spans() == ()


async def test_stream_records_thinking_duration(
    otel_setup, make_sse_event, make_sse_resp
):
    exporter, _ = otel_setup
    thoughts: list[str] = []
    gemini = GeminiClient(
        model="gemini-2.5-flash", api_key="test", on_thought=thoughts.append
    )
    original_cb = gemini.on_thought
    gemini._session = AsyncMock()
    chunks = [
        {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}]}}]},
        {
            "candidates": [
                {"content": {"parts": [{"text": "let's see", "thought": True}]}}
            ]
        },
        {
            "candidates": [
                {"content": {"parts": [{"text": "42"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {
                "totalTokenCount": 40,
                "promptTokenCount": 10,
                "candidatesTokenCount": 20,
                "thoughtsTokenCount": 10,
            },
        },
    ]
    events = [make_sse_event(_json_dumps(c)) for c in chunks]
    gemini._session.post.return_value = make_sse_resp(events)

    stream = gemini.stream_chat([{"role": "user", "content": "meaning of life?"}])
    text = [t async for t in stream]

    assert text == ["42"]
    assert thoughts == ["hmm", "let's see"]
    assert gemini.on_thought is original_cb
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 10
    assert attrs["padwan_llm.thinking.duration"] >= 0


async def test_agent_tool_execution_emits_span(otel_setup):
    exporter, reader = otel_setup

    async def echo(args: dict) -> str:
        return args["text"]

    tool = McpTool(
        name="echo",
        description="echo",
        input_schema={"type": "object"},
        handler=echo,
    )
    session, _ = make_session(
        [
            FakeChatStream(
                chunks=[], tool_calls=[make_tool_call("echo", {"text": "hi"})]
            ),
            FakeChatStream(
                chunks=["done"], usage={"total": 30, "input": 10, "output": 20}
            ),
        ],
        mcp_tools=[tool],
    )

    out = await session.send("run echo")

    assert out == "done"
    tool_span, agent_span = exporter.get_finished_spans()
    assert tool_span.name == "execute_tool echo"
    attrs = dict(tool_span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["gen_ai.tool.name"] == "echo"
    assert attrs["gen_ai.tool.call.id"] == "call_1"
    assert attrs["gen_ai.tool.description"] == "echo"
    assert attrs["gen_ai.conversation.id"] == session.session_id
    assert agent_span.name == "invoke_agent"
    agent_attrs = dict(agent_span.attributes or {})
    assert agent_attrs["gen_ai.operation.name"] == "invoke_agent"
    assert agent_attrs["gen_ai.conversation.id"] == session.session_id
    assert agent_attrs["gen_ai.usage.input_tokens"] == 10
    assert agent_attrs["gen_ai.usage.output_tokens"] == 20
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert {
        "gen_ai.invoke_agent.duration",
        "gen_ai.invoke_agent.inference_calls",
        "gen_ai.invoke_agent.tool_calls",
        "gen_ai.execute_tool.duration",
    } <= _metric_names(reader)
    assert _histogram_sum(reader, "gen_ai.invoke_agent.inference_calls") == 2
    assert _histogram_sum(reader, "gen_ai.invoke_agent.tool_calls") == 1


async def test_embeddings_span(otel_setup, make_resp):
    exporter, reader = otel_setup
    mistral = MistralClient(api_key="test")
    mistral._session = AsyncMock()
    payload = {
        "id": "emb_1",
        "object": "list",
        "model": "mistral-embed",
        "data": [{"embedding": [0.1, 0.2], "index": 0}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 0, "total_tokens": 4},
    }
    mistral._session.post.return_value = make_resp(200, payload)

    await mistral.fetch_embeddings("hello")

    (span,) = exporter.get_finished_spans()
    assert span.name == "embeddings mistral-embed"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "embeddings"
    assert attrs["gen_ai.request.model"] == "mistral-embed"
    assert "gen_ai.client.operation.duration" in _metric_names(reader)


async def test_batch_operation_span(otel_setup, client, make_resp):
    exporter, _ = otel_setup
    client._session.get.return_value = make_resp(
        200, {"id": "batch_1", "status": "completed"}
    )

    await client.get_batch("batch_1")

    (span,) = exporter.get_finished_spans()
    assert span.name == "get_batch"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "get_batch"
    assert "gen_ai.request.model" not in attrs


@dataclass
class FakeRealtimeClient(RealtimeClientBase[str]):
    provider: ClassVar[Provider] = "openai"
    model: str = "rt-mini"
    base_url: str = field(default="wss://rt.example.com/v1", kw_only=True)
    connect_error: Exception | None = None

    def _get_default_api_key(self) -> str:
        return "k"

    def _set_auth_headers(self, session) -> None:
        pass

    def _connect(self):
        @asynccontextmanager
        async def cm():
            if self.connect_error is not None:
                raise self.connect_error
            yield "conn"

        return cm()


@pytest.mark.parametrize(
    ("connect_error", "expectation", "expected_status"),
    [
        pytest.param(None, nullcontext(), StatusCode.UNSET, id="session"),
        pytest.param(
            ValueError("nope"),
            pytest.raises(ValueError),
            StatusCode.ERROR,
            id="connect_failure",
        ),
    ],
)
async def test_realtime_session_span(
    otel_setup,
    connect_error: Exception | None,
    expectation,
    expected_status: StatusCode,
):
    exporter, _ = otel_setup

    with expectation:
        async with FakeRealtimeClient(connect_error=connect_error) as conn:
            assert conn == "conn"
            assert exporter.get_finished_spans() == ()

    (span,) = exporter.get_finished_spans()
    assert span.name == "realtime rt-mini"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "realtime"
    assert attrs["server.address"] == "rt.example.com"
    assert span.status.status_code == expected_status
    if connect_error is not None:
        assert attrs["error.type"] == "ValueError"


async def test_mcp_tool_call_emits_span(otel_setup):
    exporter, reader = otel_setup
    mcp = McpStreamable(url="https://mcp.example.com/mcp")
    mcp._rpc = AsyncMock(  # type: ignore[method-assign]
        return_value={"content": [{"type": "text", "text": "ok"}]}
    )

    await mcp._call("search", {"q": "x"})

    (span,) = exporter.get_finished_spans()
    assert span.name == "tools/call search"
    attrs = dict(span.attributes or {})
    assert attrs["mcp.method.name"] == "tools/call"
    assert attrs["mcp.protocol.version"]
    assert attrs["gen_ai.tool.name"] == "search"
    assert attrs["gen_ai.operation.name"] == "execute_tool"
    assert attrs["network.transport"] == "tcp"
    assert attrs["network.protocol.name"] == "http"
    assert attrs["server.address"] == "mcp.example.com"
    assert attrs["server.port"] == 443
    assert "mcp.client.operation.duration" in _metric_names(reader)


async def test_agent_mcp_call_enriches_tool_span_without_duplicate(otel_setup):
    exporter, reader = otel_setup
    mcp = McpStreamable(url="https://mcp.example.com/mcp")
    mcp._rpc = AsyncMock(  # type: ignore[method-assign]
        return_value={"content": [{"type": "text", "text": "ok"}]}
    )
    tool = McpTool(
        name="search",
        description="Search",
        input_schema={"type": "object"},
        handler=lambda args: mcp._call("search", args),
    )
    session, _ = make_session(
        [
            FakeChatStream(
                chunks=[], tool_calls=[make_tool_call("search", {"q": "x"})]
            ),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=[tool],
    )

    assert await session.send("search") == "done"

    tool_span, agent_span = exporter.get_finished_spans()
    assert tool_span.name == "execute_tool search"
    attrs = dict(tool_span.attributes or {})
    assert attrs["mcp.method.name"] == "tools/call"
    assert attrs["server.address"] == "mcp.example.com"
    assert agent_span.name == "invoke_agent"
    assert "mcp.client.operation.duration" in _metric_names(reader)


@pytest.mark.parametrize(
    ("streaming", "payloads", "expected_reasons", "expected_tools"),
    [
        pytest.param(
            False,
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"function": {"name": "get_weather", "arguments": "{}"}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": USAGE,
                "service_tier": "flex",
            },
            ("tool_calls",),
            ("get_weather",),
            id="complete",
        ),
        pytest.param(
            True,
            [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"name": "get_weather"}}
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                    "usage": USAGE,
                    "service_tier": "flex",
                },
            ],
            ("tool_calls",),
            ("get_weather",),
            id="stream",
        ),
    ],
)
async def test_raw_openai_call_opens_span(
    otel_setup,
    client,
    make_resp,
    make_sse_event,
    make_sse_resp,
    streaming,
    payloads,
    expected_reasons,
    expected_tools,
):
    """Raw `complete()`/`stream()` outside the chat API still produce a span."""
    exporter, reader = otel_setup
    body = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hey"}],
        "service_tier": "flex",
    }
    if streaming:
        events = [make_sse_event(_json_dumps(c)) for c in payloads]
        client._session.post.return_value = make_sse_resp(events)
        chunks = [c async for c in client.stream(body)]
        assert len(chunks) == len(payloads)
    else:
        client._session.post.return_value = make_resp(200, payloads)
        await client.complete(body)

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o-mini"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.request.model"] == "gpt-4o-mini"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["gen_ai.response.finish_reasons"] == expected_reasons
    assert attrs["padwan_llm.response.tool_names"] == expected_tools
    assert attrs["openai.request.service_tier"] == "flex"
    assert attrs["openai.response.service_tier"] == "flex"
    assert attrs.get("gen_ai.request.stream", False) is streaming
    assert {"gen_ai.client.operation.duration", "gen_ai.client.token.usage"} <= (
        _metric_names(reader)
    )


async def test_raw_openai_error_ends_span(otel_setup, client, make_resp):
    exporter, _ = otel_setup
    client._session.post.return_value = make_resp(500, {"error": "boom"})
    with pytest.raises(LLMError):
        await client.complete({"model": "gpt-4o", "messages": []})
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "LLMError"


@pytest.fixture
def otel_logging():
    """Instrument with in-memory span and log exporters via a capture-aware factory."""
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    log_exporter = InMemoryLogRecordExporter()
    logger_provider = LoggerProvider()
    logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

    def _instrument(*, capture_content: bool = True):
        otel.instrument(
            tracer_provider=tracer_provider,
            meter_provider=MeterProvider(metric_readers=[InMemoryMetricReader()]),
            logger_provider=logger_provider,
            capture_content=capture_content,
        )
        return span_exporter, log_exporter

    yield _instrument
    otel.uninstrument()


async def test_capture_content_on_complete(otel_logging, client, make_resp):
    otel_capture = otel_logging()
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)

    tool = {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {}},
    }
    await client.complete_chat(
        [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hey"},
        ],
        tools=[tool],
    )

    span_exporter, log_exporter = otel_capture
    (span,) = span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert _json_loads(attrs["gen_ai.input.messages"]) == [
        {"role": "system", "parts": [{"type": "text", "content": "be brief"}]},
        {"role": "user", "parts": [{"type": "text", "content": "hey"}]},
    ]
    assert "gen_ai.system_instructions" not in attrs
    assert _json_loads(attrs["gen_ai.tool.definitions"]) == [
        {"type": "function", **tool}
    ]
    assert _json_loads(attrs["gen_ai.output.messages"]) == [
        {
            "role": "assistant",
            "parts": [{"type": "text", "content": "hi"}],
            "finish_reason": "stop",
        }
    ]
    (log,) = log_exporter.get_finished_logs()
    record = log.log_record
    assert record.event_name == "gen_ai.client.inference.operation.details"
    detail_attrs = dict(record.attributes or {})
    assert detail_attrs["gen_ai.input.messages"][0]["role"] == "system"
    assert detail_attrs["gen_ai.tool.definitions"][0]["name"] == "get_weather"


async def test_capture_content_strips_binary_parts(otel_logging, client, make_resp):
    span_exporter, _ = otel_logging()
    payload = {
        "choices": [{"message": {"content": "a duck"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)

    await client.complete_chat(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,UElORw=="},
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "UklGRg==", "format": "wav"},
                    },
                ],
            }
        ]
    )

    (span,) = span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert _json_loads(attrs["gen_ai.input.messages"]) == [
        {
            "role": "user",
            "parts": [
                {"type": "text", "content": "what is this?"},
                {"type": "image_url"},
                {"type": "input_audio"},
            ],
        }
    ]
    for value in attrs.values():
        if isinstance(value, str):
            assert "UklGRg==" not in value
            assert "UElORw==" not in value


async def test_capture_content_on_stream(
    otel_logging, client, make_sse_event, make_sse_resp
):
    otel_capture = otel_logging()
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
            "usage": USAGE,
        },
    ]
    events = [make_sse_event(_json_dumps(c)) for c in chunks]
    client._session.post.return_value = make_sse_resp(events)

    _ = [t async for t in client.stream_chat([{"role": "user", "content": "hey"}])]

    span_exporter, _ = otel_capture
    (span,) = span_exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert _json_loads(attrs["gen_ai.input.messages"]) == [
        {"role": "user", "parts": [{"type": "text", "content": "hey"}]}
    ]
    (out,) = _json_loads(attrs["gen_ai.output.messages"])
    assert out["parts"] == [{"type": "text", "content": "Hello world"}]


async def test_content_not_captured_by_default(otel_setup, client, make_resp):
    exporter, _ = otel_setup
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)

    await client.complete_chat([{"role": "user", "content": "hey"}])

    (span,) = exporter.get_finished_spans()
    assert "gen_ai.input.messages" not in dict(span.attributes or {})


async def test_error_emits_exception_event(otel_logging):
    span_exporter, log_exporter = otel_logging(capture_content=False)
    c = OpenAIClient(model=None, api_key="test")

    with pytest.raises(LLMError):
        await c.complete_chat([{"role": "user", "content": "hey"}])

    (log,) = log_exporter.get_finished_logs()
    record = log.log_record
    assert record.event_name == "gen_ai.client.operation.exception"
    assert record.severity_number == SeverityNumber.WARN
    attrs = dict(record.attributes or {})
    assert attrs["error.type"] == "LLMError"
    assert "LLMError" in str(attrs["exception.type"])
    (span,) = span_exporter.get_finished_spans()
    assert span.context is not None
    assert record.trace_id == span.context.trace_id


# semconv advised boundaries; the SDK default ladder would collapse GenAI latencies into one bucket
_SEMCONV_DURATION = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)
_SEMCONV_TOKENS = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
_SEMCONV_AGENT_DURATION = (
    0.1,
    0.2,
    0.4,
    0.8,
    1.6,
    3.2,
    6.4,
    12.8,
    25.6,
    51.2,
    102.4,
    204.8,
    409.6,
)
_SEMCONV_AGENT_CALLS = (1, 2, 4, 8, 16, 32, 64, 128)


@pytest.fixture
async def recorded_histograms(
    otel_setup, client, make_resp, make_sse_event, make_sse_resp
) -> InMemoryMetricReader:
    """Drive a chat, a stream and an agent turn so every histogram has a data point."""
    _, reader = otel_setup
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)
    await client.complete_chat([{"role": "user", "content": "hey"}])

    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
            "usage": USAGE,
        },
    ]
    client._session.post.return_value = make_sse_resp(
        [make_sse_event(_json_dumps(c)) for c in chunks]
    )
    async for _ in client.stream_chat([{"role": "user", "content": "hey"}]):
        pass

    async def echo(args: dict) -> str:
        return args["text"]

    session, _ = make_session(
        [
            FakeChatStream(
                chunks=[], tool_calls=[make_tool_call("echo", {"text": "hi"})]
            ),
            FakeChatStream(
                chunks=["done"], usage={"total": 30, "input": 10, "output": 20}
            ),
        ],
        mcp_tools=[
            McpTool(
                name="echo",
                description="echo",
                input_schema={"type": "object"},
                handler=echo,
            )
        ],
    )
    await session.send("run echo")
    return reader


@pytest.mark.parametrize(
    ("metric", "boundaries"),
    [
        pytest.param(
            "gen_ai.client.operation.duration", _SEMCONV_DURATION, id="duration"
        ),
        pytest.param("gen_ai.client.token.usage", _SEMCONV_TOKENS, id="tokens"),
        pytest.param(
            "gen_ai.client.operation.time_to_first_chunk", _SEMCONV_DURATION, id="ttfc"
        ),
        pytest.param(
            "gen_ai.client.operation.time_per_output_chunk",
            _SEMCONV_DURATION,
            id="per_chunk",
        ),
        pytest.param(
            "gen_ai.invoke_agent.duration", _SEMCONV_AGENT_DURATION, id="agent"
        ),
        pytest.param(
            "gen_ai.invoke_agent.inference_calls",
            _SEMCONV_AGENT_CALLS,
            id="inference_calls",
        ),
        pytest.param(
            "gen_ai.invoke_agent.tool_calls", _SEMCONV_AGENT_CALLS, id="tool_calls"
        ),
        pytest.param(
            "gen_ai.execute_tool.duration", _SEMCONV_DURATION, id="execute_tool"
        ),
    ],
)
async def test_histogram_buckets_follow_semconv(
    recorded_histograms: InMemoryMetricReader,
    metric: str,
    boundaries: tuple[float, ...],
) -> None:
    points = _histogram_points(recorded_histograms, metric)
    assert points, f"{metric} recorded no data point"
    for point in points:
        assert tuple(point.explicit_bounds) == boundaries
