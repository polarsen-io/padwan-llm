import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable, Sequence
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
from ._base import ChatStream, LLMClientBase, RealtimeClientBase
from .agent import AgentSession
from .anthropic import AnthropicClient
from .errors import Provider
from .gemini import GeminiClient
from .grok import GrokClient
from .mistral import MistralClient
from .models import ToolCall, UsageToken
from .openai import OpenAIClient

__all__ = ("instrument", "uninstrument")

# GenAI semconv attributes are inlined as literals to avoid the incubating
# semconv package.

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

# batch/file operations instrumented as plain client spans, per defining class
_BATCH_METHODS: dict[type[LLMClientBase], tuple[str, ...]] = {
    OpenAIClient: (
        "upload_batch_file",
        "create_batch",
        "get_batch",
        "list_batches",
        "cancel_batch",
        "get_batch_results",
    ),
    GeminiClient: ("create_batch", "get_batch", "list_batches", "cancel_batch"),
    GrokClient: (
        "create_batch",
        "add_batch_requests",
        "get_batch",
        "list_batches",
        "cancel_batch",
        "get_batch_results",
    ),
}

_originals: dict[type, dict[str, Any]] = {}


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
        methods: dict[str, Any] = {
            "complete_chat": cls.complete_chat,
            "stream_chat": cls.stream_chat,
        }
        setattr(cls, "complete_chat", _wrap_complete_chat(cls.complete_chat, inst))
        setattr(cls, "stream_chat", _wrap_stream_chat(cls.stream_chat, inst))
        for name in _BATCH_METHODS.get(cls, ()):
            methods[name] = getattr(cls, name)
            setattr(cls, name, _wrap_operation(methods[name], inst, name))
        _originals[cls] = methods

    _originals[MistralClient]["fetch_embeddings"] = MistralClient.fetch_embeddings
    setattr(
        MistralClient,
        "fetch_embeddings",
        _wrap_operation(
            MistralClient.fetch_embeddings, inst, "embeddings", model_param="model"
        ),
    )

    _originals[AgentSession] = {"_dispatch_one": AgentSession._dispatch_one}
    setattr(
        AgentSession,
        "_dispatch_one",
        _wrap_execute_tool(AgentSession._dispatch_one, inst),
    )

    _originals[RealtimeClientBase] = {
        "__aenter__": RealtimeClientBase.__aenter__,
        "__aexit__": RealtimeClientBase.__aexit__,
    }
    setattr(
        RealtimeClientBase,
        "__aenter__",
        _wrap_realtime_enter(RealtimeClientBase.__aenter__, inst),
    )
    setattr(
        RealtimeClientBase,
        "__aexit__",
        _wrap_realtime_exit(RealtimeClientBase.__aexit__, inst),
    )


def uninstrument() -> None:
    """Restore the original client methods."""
    for cls, methods in _originals.items():
        for name, fn in methods.items():
            setattr(cls, name, fn)
    _originals.clear()


def _request_attrs(
    client: LLMClientBase | RealtimeClientBase[Any], op: str = "chat"
) -> dict[str, Any]:
    """Low-cardinality attributes shared by the span and both metrics."""
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": op,
        "gen_ai.provider.name": _PROVIDER_NAMES.get(client.provider, client.provider),
    }
    if client.model:
        attrs["gen_ai.request.model"] = client.model
    if host := urlsplit(client.base_url).hostname:
        attrs["server.address"] = host
    return attrs


class _ThoughtTimer:
    """Chains to the client's on_thought callback, timestamping thought chunks."""

    def __init__(self, wrapped: Callable[[str], None] | None) -> None:
        self.wrapped = wrapped
        self.first: float | None = None
        self.last: float | None = None

    def __call__(self, text: str) -> None:
        now = time.perf_counter()
        if self.first is None:
            self.first = now
        self.last = now
        if self.wrapped is not None:
            self.wrapped(text)


def _tool_names(tool_calls: Sequence[ToolCall] | None) -> tuple[str, ...] | None:
    return tuple(tc["function"]["name"] for tc in tool_calls) if tool_calls else None


def _record_end(
    inst: _Instruments,
    span: trace.Span,
    attrs: dict[str, Any],
    start: float,
    *,
    usage: UsageToken | None = None,
    finish_reasons: list[str] | None = None,
    tool_names: tuple[str, ...] | None = None,
    thinking: _ThoughtTimer | None = None,
    error: BaseException | None = None,
) -> None:
    elapsed = time.perf_counter() - start
    metric_attrs = dict(attrs)
    if error is not None:
        error_type = type(error).__qualname__
        metric_attrs["error.type"] = error_type
        span.set_attribute("error.type", error_type)
        span.set_status(StatusCode.ERROR, str(error))
        span.record_exception(error)
    inst.duration.record(elapsed, metric_attrs)
    if usage is not None:
        span.set_attribute("gen_ai.usage.input_tokens", usage["input"])
        span.set_attribute("gen_ai.usage.output_tokens", usage["output"])
        if (reasoning := usage.get("reasoning")) is not None:
            span.set_attribute("gen_ai.usage.reasoning_tokens", reasoning)
        for token_type, count in (
            ("input", usage["input"]),
            ("output", usage["output"]),
        ):
            inst.tokens.record(count, {**metric_attrs, "gen_ai.token.type": token_type})
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)
    if tool_names:
        span.set_attribute("padwan_llm.response.tool_names", tool_names)
    if (
        thinking is not None
        and thinking.first is not None
        and thinking.last is not None
    ):
        span.set_attribute(
            "padwan_llm.thinking.duration", thinking.last - thinking.first
        )
    span.end()


def _sent_temperature(client: LLMClientBase) -> float | None:
    """Temperature actually sent on the wire; AnthropicClient never sends one."""
    return None if client.provider == "anthropic" else client.temperature


def _start_span(
    inst: _Instruments, attrs: dict[str, Any], temperature: float | None
) -> trace.Span:
    model = attrs.get("gen_ai.request.model")
    span_attrs = dict(attrs)
    if temperature is not None:
        span_attrs["gen_ai.request.temperature"] = temperature
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
            tool_names=_tool_names(response.get("tool_calls")),
        )
        return response, usage

    return wrapper


def _wrap_stream_chat(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    def wrapper(self: LLMClientBase, *args: Any, **kwargs: Any) -> ChatStream:
        # Swap on_thought around the (synchronous) stream construction so the
        # provider stream captures the timing callback.
        thinking = _ThoughtTimer(self.on_thought)
        self.on_thought = thinking
        try:
            stream: ChatStream = original(self, *args, **kwargs)
        finally:
            self.on_thought = thinking.wrapped
        return _InstrumentedChatStream(
            stream, inst, _request_attrs(self), _sent_temperature(self), thinking
        )

    return wrapper


def _wrap_operation(
    original: Any, inst: _Instruments, op: str, model_param: str | None = None
) -> Any:
    """Wrap a plain async client method in a CLIENT span named after `op`.

    Batch operations are not tied to the client's default model, so the model
    attribute is only set when `model_param` names the method's model argument.
    """
    sig = inspect.signature(original) if model_param else None

    @functools.wraps(original)
    async def wrapper(self: LLMClientBase, *args: Any, **kwargs: Any) -> Any:
        attrs = _request_attrs(self, op=op)
        attrs.pop("gen_ai.request.model", None)
        if sig is not None and model_param is not None:
            bound = sig.bind(self, *args, **kwargs)
            bound.apply_defaults()
            attrs["gen_ai.request.model"] = bound.arguments[model_param]
        model = attrs.get("gen_ai.request.model")
        span = inst.tracer.start_span(
            f"{op} {model}" if model else op, kind=SpanKind.CLIENT, attributes=attrs
        )
        start = time.perf_counter()
        try:
            result = await original(self, *args, **kwargs)
        except BaseException as e:
            _record_end(inst, span, attrs, start, error=e)
            raise
        _record_end(inst, span, attrs, start)
        return result

    return wrapper


def _wrap_execute_tool(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(
        self: AgentSession, tc: ToolCall, *args: Any, **kwargs: Any
    ) -> str:
        name = tc["function"]["name"]
        span = inst.tracer.start_span(
            f"execute_tool {name}",
            kind=SpanKind.INTERNAL,
            attributes={
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": name,
                "gen_ai.tool.type": "function",
                "gen_ai.tool.call.id": tc["id"],
            },
        )
        with trace.use_span(
            span,
            end_on_exit=True,
            record_exception=True,
            set_status_on_exception=True,
        ):
            return await original(self, tc, *args, **kwargs)

    return wrapper


def _wrap_realtime_enter(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(self: Any) -> Any:
        attrs = _request_attrs(self, op="realtime")
        span = inst.tracer.start_span(
            f"realtime {self.model}", kind=SpanKind.CLIENT, attributes=attrs
        )
        start = time.perf_counter()
        try:
            conn = await original(self)
        except BaseException as e:
            _record_end(inst, span, attrs, start, error=e)
            raise
        self._otel_session = (span, attrs, start)
        return conn

    return wrapper


def _wrap_realtime_exit(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(self: Any, *args: Any) -> None:
        try:
            return await original(self, *args)
        finally:
            if session := getattr(self, "_otel_session", None):
                self._otel_session = None
                span, attrs, start = session
                exc_val = args[1] if len(args) > 1 else None
                _record_end(inst, span, attrs, start, error=exc_val)

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
        thinking: _ThoughtTimer | None = None,
    ) -> None:
        self._inner = inner
        self._inst = inst
        self._attrs = attrs
        self._temperature = temperature
        self._thinking = thinking

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
                    tool_names=_tool_names(self._inner.tool_calls),
                    thinking=self._thinking,
                )
