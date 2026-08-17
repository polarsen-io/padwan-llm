import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from padwan_llm import otel
from padwan_llm.errors import LLMError
from padwan_llm.openai import OpenAIClient

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


async def test_complete_chat_emits_span_and_metrics(otel_setup, client, make_resp):
    exporter, reader = otel_setup
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)

    await client.complete_chat([{"role": "user", "content": "hey"}])

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat gpt-4o"
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.provider.name"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["server.address"] == "api.openai.com"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 20
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)
    assert _metric_names(reader) == {
        "gen_ai.client.operation.duration",
        "gen_ai.client.token.usage",
    }


async def test_stream_chat_span_ends_after_iteration(
    otel_setup, client, make_sse_event, make_sse_resp
):
    exporter, _ = otel_setup
    chunks = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {
            "choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}],
            "usage": USAGE,
        },
    ]
    events = [make_sse_event(json.dumps(c)) for c in chunks]
    client._session.post.return_value = make_sse_resp(events)

    stream = client.stream_chat([{"role": "user", "content": "hey"}])
    assert exporter.get_finished_spans() == ()

    text = [t async for t in stream]

    assert "".join(text) == "Hello world"
    assert stream.usage == {"total": 30, "input": 10, "output": 20}
    (span,) = exporter.get_finished_spans()
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.response.finish_reasons"] == ("stop",)


async def test_error_records_error_type(otel_setup):
    exporter, _ = otel_setup
    c = OpenAIClient(model=None, api_key="test")

    with pytest.raises(LLMError):
        await c.complete_chat([{"role": "user", "content": "hey"}])

    (span,) = exporter.get_finished_spans()
    assert span.name == "chat"
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "LLMError"


async def test_complete_chat_cancellation_ends_span(otel_setup, client):
    exporter, _ = otel_setup
    client._session.post.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await client.complete_chat([{"role": "user", "content": "hey"}])

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert dict(span.attributes or {})["error.type"] == "CancelledError"


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
        make_sse_event(json.dumps({"choices": [{"delta": {"content": t}}]}))
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


async def test_instrument_is_idempotent(otel_setup, client, make_resp):
    exporter, _ = otel_setup
    otel.instrument()
    payload = {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": USAGE,
    }
    client._session.post.return_value = make_resp(200, payload)

    await client.complete_chat([{"role": "user", "content": "hey"}])

    assert len(exporter.get_finished_spans()) == 1


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
