import functools
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError as e:
    raise ImportError(
        "padwan_llm.otel requires OpenTelemetry: pip install 'padwan-llm[otel]'"
    ) from e

from . import __version__
from ._base import ChatStream, LLMClientBase
from .anthropic import AnthropicClient
from .errors import Provider
from .gemini import GeminiClient
from .grok import GrokClient
from .mistral import MistralClient
from .models import UsageToken
from .openai import OpenAIClient

__all__ = ("instrument", "uninstrument")

# OTel GenAI semconv attribute names (pre-stable), inlined to avoid the
# incubating semconv package.
_OPERATION = "gen_ai.operation.name"
_PROVIDER = "gen_ai.provider.name"
_REQ_MODEL = "gen_ai.request.model"
_REQ_TEMPERATURE = "gen_ai.request.temperature"
_RESP_FINISH = "gen_ai.response.finish_reasons"
_USAGE_INPUT = "gen_ai.usage.input_tokens"
_USAGE_OUTPUT = "gen_ai.usage.output_tokens"
_TOKEN_TYPE = "gen_ai.token.type"
_ERROR_TYPE = "error.type"
_SERVER_ADDRESS = "server.address"

# semconv well-known values for gen_ai.provider.name
_PROVIDER_NAMES: dict[Provider, str] = {
    "openai": "openai",
    "gemini": "gcp.gemini",
    "mistral": "mistral_ai",
    "grok": "x_ai",
    "anthropic": "anthropic",
}

_CLIENT_CLASSES: tuple[type[LLMClientBase], ...] = (
    OpenAIClient,
    GeminiClient,
    MistralClient,
    GrokClient,
    AnthropicClient,
)

_originals: dict[type[LLMClientBase], dict[str, Any]] = {}


@dataclass
class _Instruments:
    tracer: trace.Tracer
    duration: metrics.Histogram
    tokens: metrics.Histogram


def instrument(
    tracer_provider: trace.TracerProvider | None = None,
    meter_provider: metrics.MeterProvider | None = None,
) -> None:
    """Emit OTel GenAI spans and metrics for all provider clients. Idempotent."""
    if _originals:
        return
    tracer = trace.get_tracer("padwan_llm", __version__, tracer_provider)
    meter = metrics.get_meter("padwan_llm", __version__, meter_provider)
    inst = _Instruments(
        tracer=tracer,
        duration=meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="Duration of chat completion operations",
        ),
        tokens=meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
        ),
    )
    for cls in _CLIENT_CLASSES:
        _originals[cls] = {
            "complete_chat": cls.complete_chat,
            "stream_chat": cls.stream_chat,
        }
        setattr(cls, "complete_chat", _wrap_complete_chat(cls.complete_chat, inst))
        setattr(cls, "stream_chat", _wrap_stream_chat(cls.stream_chat, inst))


def uninstrument() -> None:
    """Restore the original client methods."""
    for cls, methods in _originals.items():
        for name, fn in methods.items():
            setattr(cls, name, fn)
    _originals.clear()


def _request_attrs(client: LLMClientBase) -> dict[str, Any]:
    """Low-cardinality attributes shared by the span and both metrics."""
    attrs: dict[str, Any] = {
        _OPERATION: "chat",
        _PROVIDER: _PROVIDER_NAMES.get(client.provider, client.provider),
    }
    if client.model:
        attrs[_REQ_MODEL] = client.model
    if host := urlsplit(client.base_url).hostname:
        attrs[_SERVER_ADDRESS] = host
    return attrs


def _record_end(
    inst: _Instruments,
    span: trace.Span,
    attrs: dict[str, Any],
    start: float,
    *,
    usage: UsageToken | None = None,
    finish_reasons: list[str] | None = None,
    error: BaseException | None = None,
) -> None:
    elapsed = time.perf_counter() - start
    metric_attrs = dict(attrs)
    if error is not None:
        error_type = type(error).__qualname__
        metric_attrs[_ERROR_TYPE] = error_type
        span.set_attribute(_ERROR_TYPE, error_type)
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)
    inst.duration.record(elapsed, metric_attrs)
    if usage is not None:
        span.set_attribute(_USAGE_INPUT, usage["input"])
        span.set_attribute(_USAGE_OUTPUT, usage["output"])
        inst.tokens.record(usage["input"], {**metric_attrs, _TOKEN_TYPE: "input"})
        inst.tokens.record(usage["output"], {**metric_attrs, _TOKEN_TYPE: "output"})
    if finish_reasons:
        span.set_attribute(_RESP_FINISH, finish_reasons)
    span.end()


def _sent_temperature(client: LLMClientBase) -> float | None:
    """Temperature actually sent on the wire; AnthropicClient never sends one."""
    return None if client.provider == "anthropic" else client.temperature


def _start_span(
    inst: _Instruments, attrs: dict[str, Any], temperature: float | None
) -> trace.Span:
    model = attrs.get(_REQ_MODEL)
    span_attrs = dict(attrs)
    if temperature is not None:
        span_attrs[_REQ_TEMPERATURE] = temperature
    return inst.tracer.start_span(
        f"chat {model}" if model else "chat",
        kind=SpanKind.CLIENT,
        attributes=span_attrs,
    )


def _wrap_complete_chat(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(self: LLMClientBase, *args: Any, **kwargs: Any) -> Any:
        attrs = _request_attrs(self)
        span = _start_span(inst, attrs, _sent_temperature(self))
        start = time.perf_counter()
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                response, usage = await original(self, *args, **kwargs)
            except BaseException as e:
                # BaseException so a cancelled task still ends the span.
                _record_end(inst, span, attrs, start, error=e)
                raise
        _record_end(
            inst,
            span,
            attrs,
            start,
            usage=usage,
            finish_reasons=[response["finish_reason"]],
        )
        return response, usage

    return wrapper


def _wrap_stream_chat(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    def wrapper(self: LLMClientBase, *args: Any, **kwargs: Any) -> ChatStream:
        stream: ChatStream = original(self, *args, **kwargs)
        return _InstrumentedChatStream(
            stream, inst, _request_attrs(self), _sent_temperature(self)
        )

    return wrapper


class _InstrumentedChatStream(ChatStream):
    """Delegates to the provider stream; the span ends when iteration completes,
    errors, or is cancelled. Abandoned streams end at generator close/GC."""

    def __init__(
        self,
        inner: ChatStream,
        inst: _Instruments,
        attrs: dict[str, Any],
        temperature: float | None,
    ) -> None:
        self._inner = inner
        self._inst = inst
        self._attrs = attrs
        self._temperature = temperature

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":  # guard against recursion before __init__ ran
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        span = _start_span(self._inst, self._attrs, self._temperature)
        start = time.perf_counter()
        ended = False
        try:
            async for chunk in self._inner:
                yield chunk
        except GeneratorExit:
            # abandoned stream: end cleanly via finally, no error status
            raise
        except BaseException as e:
            # BaseException so cancellation records error.type, not success.
            _record_end(self._inst, span, self._attrs, start, error=e)
            ended = True
            raise
        finally:
            # finally (not else) so an abandoned stream (GeneratorExit) still
            # ends the span.
            self.usage = self._inner.usage
            self.tool_calls = self._inner.tool_calls
            if not ended:
                reason = getattr(self._inner, "finish_reason", None)
                _record_end(
                    self._inst,
                    span,
                    self._attrs,
                    start,
                    usage=self._inner.usage,
                    finish_reasons=[reason] if reason else None,
                )
