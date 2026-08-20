from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Self

try:
    from langfuse import Langfuse
    from langfuse.span_filter import is_default_export_span
    from langfuse.types import (
        MaskOtelSpansFunction,
        MaskOtelSpansParams,
        MaskOtelSpansResult,
        OtelSpanData,
        OtelSpanIdentifier,
        OtelSpanPatch,
    )
    from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
    from opentelemetry.util.types import AttributeValue
except ImportError as e:
    raise ImportError(
        "padwan_llm.langfuse requires Langfuse: pip install 'padwan-llm[langfuse]'"
    ) from e

from . import otel
from ._json import dumps as _json_dumps, loads as _json_loads

__all__ = ("LangfuseIntegration", "instrument")

_PADWAN_SCOPE = "padwan_llm"
_OBSERVATION_TYPES = {
    "chat": "generation",
    "embeddings": "embedding",
    "execute_tool": "tool",
    "invoke_agent": "agent",
}


def _string_attribute(
    attributes: Mapping[str, AttributeValue], name: str
) -> str | None:
    value = attributes.get(name)
    return value if isinstance(value, str) else None


def _decode_json(value: str) -> object:
    try:
        return _json_loads(value)
    except ValueError:
        return value


def _input_attribute(attributes: Mapping[str, AttributeValue]) -> str | None:
    messages = _string_attribute(attributes, "gen_ai.input.messages")
    system = _string_attribute(attributes, "gen_ai.system_instructions")
    tools = _string_attribute(attributes, "gen_ai.tool.definitions")
    if messages is not None and system is None and tools is None:
        return messages
    sections: dict[str, object] = {}
    if messages is not None:
        sections["messages"] = _decode_json(messages)
    if system is not None:
        sections["system_instructions"] = _decode_json(system)
    if tools is not None:
        sections["tools"] = _decode_json(tools)
    if sections:
        return _json_dumps(sections)
    return _string_attribute(attributes, "gen_ai.tool.call.arguments")


def _output_attribute(attributes: Mapping[str, AttributeValue]) -> str | None:
    return _string_attribute(attributes, "gen_ai.output.messages") or _string_attribute(
        attributes, "gen_ai.tool.call.result"
    )


def _mapped_attributes(
    span: OtelSpanData, attributes: Mapping[str, AttributeValue]
) -> dict[str, AttributeValue]:
    if span.instrumentation_scope_name != _PADWAN_SCOPE:
        return {}
    mapped: dict[str, AttributeValue] = {}
    operation = _string_attribute(attributes, "gen_ai.operation.name")
    values: tuple[tuple[str, AttributeValue | None], ...] = (
        (
            "langfuse.observation.type",
            _OBSERVATION_TYPES.get(operation or "", "span"),
        ),
        ("langfuse.observation.input", _input_attribute(attributes)),
        ("langfuse.observation.output", _output_attribute(attributes)),
        (
            "langfuse.session.id",
            _string_attribute(attributes, "gen_ai.conversation.id"),
        ),
    )
    for name, value in values:
        if value is not None and name not in attributes:
            mapped[name] = value
    return mapped


def _apply_patch(
    attributes: Mapping[str, AttributeValue], patch: OtelSpanPatch | None
) -> dict[str, AttributeValue]:
    patched = dict(attributes)
    if patch is None:
        return patched
    for name in patch.delete_attributes:
        patched.pop(name, None)
    patched.update(patch.set_attributes)
    return patched


@dataclass(frozen=True)
class _SpanAdapter:
    user_mask: MaskOtelSpansFunction | None = None

    def __call__(self, *, params: MaskOtelSpansParams) -> MaskOtelSpansResult | None:
        user_result = self.user_mask(params=params) if self.user_mask else None
        if user_result is not None and not isinstance(user_result, MaskOtelSpansResult):
            raise TypeError("mask_otel_spans must return MaskOtelSpansResult or None")
        user_patches = user_result.span_patches if user_result else {}
        patches: dict[OtelSpanIdentifier, OtelSpanPatch | None] = {}
        for identifier, span in params.spans.items():
            user_patch = user_patches.get(identifier)
            if user_patch is not None and not isinstance(user_patch, OtelSpanPatch):
                raise TypeError("mask_otel_spans contains an invalid span patch")
            mapped = _mapped_attributes(span, _apply_patch(span.attributes, user_patch))
            if user_patch is None and not mapped:
                continue
            deleted = tuple(user_patch.delete_attributes) if user_patch else ()
            for name in deleted:
                mapped.pop(name, None)
            if user_patch is not None:
                mapped.update(user_patch.set_attributes)
            patches[identifier] = OtelSpanPatch(
                set_attributes=mapped,
                delete_attributes=deleted,
            )
        for identifier, patch in user_patches.items():
            if identifier not in params.spans:
                patches[identifier] = patch
        return MaskOtelSpansResult(span_patches=patches) if patches else None


@dataclass(frozen=True)
class _SpanFilter:
    user_filter: Callable[[ReadableSpan], bool] | None = None

    def __call__(self, span: ReadableSpan) -> bool:
        scope = span.instrumentation_scope
        selected = is_default_export_span(span) or (
            scope is not None and scope.name == _PADWAN_SCOPE
        )
        return selected and (self.user_filter(span) if self.user_filter else True)


@dataclass
class LangfuseIntegration:
    """Own the Langfuse client and Padwan instrumentation lifecycle."""

    client: Langfuse
    tracer_provider: TracerProvider
    _closed: bool = field(default=False, init=False, repr=False)

    def flush(self) -> None:
        """Flush pending spans to Langfuse."""
        if self._closed:
            raise RuntimeError("Langfuse integration is shut down")
        self.client.flush()

    def shutdown(self) -> None:
        """Restore Padwan methods and stop the Langfuse client."""
        if self._closed:
            return
        try:
            otel.uninstrument()
        finally:
            try:
                self.client.shutdown()
            finally:
                self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.shutdown()


def instrument(
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    base_url: str | None = None,
    capture_content: bool = False,
    tracer_provider: TracerProvider | None = None,
    environment: str | None = None,
    release: str | None = None,
    sample_rate: float | None = None,
    timeout: int | None = None,
    flush_at: int | None = None,
    flush_interval: float | None = None,
    debug: bool = False,
    mask_otel_spans: MaskOtelSpansFunction | None = None,
    should_export_span: Callable[[ReadableSpan], bool] | None = None,
) -> LangfuseIntegration:
    """Instrument Padwan and export enriched spans through Langfuse."""
    if otel.is_instrumented():
        raise RuntimeError(
            "Padwan OpenTelemetry instrumentation is already active; "
            "call otel.uninstrument() first"
        )
    provider = tracer_provider or TracerProvider()
    client = Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        base_url=base_url,
        tracer_provider=provider,
        environment=environment,
        release=release,
        sample_rate=sample_rate,
        timeout=timeout,
        flush_at=flush_at,
        flush_interval=flush_interval,
        debug=debug,
        mask_otel_spans=_SpanAdapter(mask_otel_spans),
        should_export_span=_SpanFilter(should_export_span),
    )
    try:
        otel.instrument(tracer_provider=provider, capture_content=capture_content)
    except BaseException:
        client.shutdown()
        raise
    return LangfuseIntegration(client=client, tracer_provider=provider)
