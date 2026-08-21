import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from padwan_llm import otel
from padwan_llm.conversation import Message
from padwan_llm.models import ToolDefinition
from tests.otel_coverage import OTEL_COVERAGE_KEY, OtelCoverage, check_spans


def _skip_no_key(env_var: str) -> pytest.MarkDecorator:
    """Lazy skipif — the string condition is evaluated at test time, not import time."""
    return pytest.mark.skipif(
        f"not __import__('os').environ.get('{env_var}')",
        reason=f"{env_var} not set",
    )


skip_no_gemini = _skip_no_key("GEMINI_API_KEY")
skip_no_openai = _skip_no_key("OPENAI_API_KEY")
skip_no_mistral = _skip_no_key("MISTRAL_API_KEY")
skip_no_grok = _skip_no_key("GROK_API_KEY")
skip_no_anthropic = _skip_no_key("ANTHROPIC_API_KEY")

pytestmark = pytest.mark.e2e

PROMPT = [Message(role="user", content="Reply with only the word 'hello'.")]

AUDIO_FIXTURE = Path(__file__).parent.parent / "fixtures" / "audio.wav"

TOOL_PROMPT = [Message(role="user", content="What is the weather in Paris?")]

WEATHER_TOOL: ToolDefinition = {
    "name": "get_weather",
    "description": "Get the current weather for a given city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "The city name"}},
        "required": ["city"],
    },
}


@pytest.fixture(scope="session")
def otel_exporter(
    request: pytest.FixtureRequest,
) -> Iterator[InMemorySpanExporter | None]:
    """Instrument the whole suite once, so every provider call under test also exercises otel.py."""
    langfuse = request.config.getoption("--langfuse")
    if not (request.config.getoption("--otel") or langfuse):
        yield None
        return
    # a readable per-run instance id; the SDK default is a UUID, which no dashboard can order
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    resource = Resource.create(
        {"service.name": "padwan-llm-e2e", "service.instance.id": run_id}
    )
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    readers: list[MetricReader] = [InMemoryMetricReader()]
    # `just e2e-otel` sets the endpoint; without it the run stays offline
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        readers.append(PeriodicExportingMetricReader(OTLPMetricExporter()))
    meter_provider = MeterProvider(resource=resource, metric_readers=readers)
    integration = None
    if langfuse:
        from padwan_llm.langfuse import instrument as langfuse_instrument

        # the adapter wires traces only, so metrics have to reach it through the global provider
        metrics.set_meter_provider(meter_provider)
        integration = langfuse_instrument(
            tracer_provider=tracer_provider, capture_content=True
        )
    else:
        otel.instrument(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            capture_content=True,
        )
    request.config.stash[OTEL_COVERAGE_KEY] = OtelCoverage()
    yield exporter
    if integration is not None:
        integration.flush()
        integration.shutdown()  # also calls otel.uninstrument()
    else:
        otel.uninstrument()
    tracer_provider.shutdown()  # flushes the batch processor
    meter_provider.shutdown()


@pytest.fixture(autouse=True)
def otel_spans(
    otel_exporter: InMemorySpanExporter | None, request: pytest.FixtureRequest
) -> Iterator[None]:
    """Check the spans a test produced, then tally them; autouse so it tears down after client fixtures."""
    if otel_exporter is None:
        yield
        return
    yield
    spans = otel_exporter.get_finished_spans()
    otel_exporter.clear()
    request.config.stash[OTEL_COVERAGE_KEY].record(spans)
    expect_error = request.node.get_closest_marker("otel_expect_error") is not None
    violations = check_spans(spans, expect_error=expect_error)
    assert not violations, "semconv violations:\n" + "\n".join(violations)
