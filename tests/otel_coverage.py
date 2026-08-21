import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

DOC_PATH = Path(__file__).parents[1] / "docs" / "observability.md"

# semconv well-known values, mirroring otel._PROVIDER_NAMES
KNOWN_PROVIDERS = frozenset({"openai", "gcp.gemini", "mistral_ai", "x_ai", "anthropic"})

# operations built from _request_attrs(): they always carry provider, model and endpoint
CLIENT_OPS = frozenset({"chat", "embeddings", "realtime"})

# attribute namespaces padwan_llm owns; anything else on a span (Langfuse span marks,
# resource attributes) belongs to a third party and is not ours to document
_NAMESPACES = "gen_ai|mcp|server|network|rpc|openai|padwan_llm|error"
_ATTRIBUTE = re.compile(rf"`((?:{_NAMESPACES})\.[a-z0-9_.]+)`")
_OWNED = re.compile(rf"^(?:{_NAMESPACES})\.")

# event names, not span attributes, but they share the gen_ai. prefix
_EVENT_NAMES = frozenset({"gen_ai.client.inference.operation.details"})


@cache
def documented_attributes(path: Path = DOC_PATH) -> frozenset[str]:
    """Span-attribute vocabulary from the docs; sections from `## Metrics` on describe metrics, events and semconv gaps."""
    body = path.read_text().split("## Metrics")[0]
    return frozenset(_ATTRIBUTE.findall(body)) - _EVENT_NAMES


def _violations(span: ReadableSpan, *, expect_error: bool) -> Iterator[str]:
    attrs: Mapping[str, Any] = span.attributes or {}
    op = attrs.get("gen_ai.operation.name")
    if op is None and "mcp.method.name" not in attrs:
        yield "neither gen_ai.operation.name nor mcp.method.name"
    provider = attrs.get("gen_ai.provider.name")
    if provider is not None and provider not in KNOWN_PROVIDERS:
        yield f"unknown gen_ai.provider.name {provider!r}"
    if op in CLIENT_OPS:
        yield from (
            f"missing {key}"
            for key in (
                "gen_ai.provider.name",
                "gen_ai.request.model",
                "server.address",
            )
            if key not in attrs
        )
    if op == "execute_tool" and "gen_ai.tool.name" not in attrs:
        yield "missing gen_ai.tool.name"
    failed = span.status.status_code is StatusCode.ERROR
    if failed and not expect_error:
        yield f"unexpected ERROR status: {span.status.description}"
    if failed and "error.type" not in attrs:
        yield "ERROR status without error.type"
    if op == "chat" and not failed:
        tokens = attrs.get("gen_ai.usage.input_tokens")
        if not isinstance(tokens, int) or tokens <= 0:
            yield f"successful chat span reporting input tokens {tokens!r}"
    if span.end_time is None:
        yield "span was never ended"


def check_spans(spans: Sequence[ReadableSpan], *, expect_error: bool) -> list[str]:
    """Baseline every emitted span must satisfy, as a list of human-readable violations."""
    return [
        f"{span.name}: {problem}"
        for span in spans
        for problem in _violations(span, expect_error=expect_error)
    ]


@dataclass
class OtelCoverage:
    spans: int = 0
    operations: set[str] = field(default_factory=set)
    attributes: set[str] = field(default_factory=set)

    def record(self, spans: Sequence[ReadableSpan]) -> None:
        for span in spans:
            attrs = span.attributes or {}
            self.spans += 1
            self.attributes.update(k for k in attrs if _OWNED.match(k))
            if op := attrs.get("gen_ai.operation.name") or attrs.get("mcp.method.name"):
                self.operations.add(str(op))

    @property
    def undocumented(self) -> set[str]:
        """Attributes the instrumentation emits that docs/observability.md never names."""
        return self.attributes - documented_attributes()

    @property
    def unobserved(self) -> frozenset[str]:
        """Documented attributes this run never saw — informational, it depends on which keys were set."""
        return documented_attributes() - self.attributes


def report(coverage: OtelCoverage) -> list[str]:
    operations = ", ".join(sorted(coverage.operations)) or "none"
    lines = [f"{coverage.spans} spans | operations: {operations}"]
    if unobserved := coverage.unobserved:
        lines.append(
            f"documented but not emitted by this run ({len(unobserved)}): "
            + ", ".join(sorted(unobserved))
        )
    if undocumented := coverage.undocumented:
        lines.append(
            f"EMITTED BUT UNDOCUMENTED ({len(undocumented)}): "
            + ", ".join(sorted(undocumented))
        )
    return lines


OTEL_COVERAGE_KEY = pytest.StashKey[OtelCoverage]()
