import functools
import inspect
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

try:
    from opentelemetry import context as otel_context, metrics, trace
    from opentelemetry._logs import Logger, LoggerProvider, SeverityNumber, get_logger
    from opentelemetry.trace import SpanKind, StatusCode
except ImportError as e:
    raise ImportError(
        "padwan_llm.otel requires OpenTelemetry: pip install 'padwan-llm[otel]'"
    ) from e

from . import __version__
from ._base import ChatStream, LLMClientBase, RealtimeClientBase
from ._json import dumps as _json_dumps, loads as _json_loads
from .agent import AgentSession
from .anthropic import AnthropicClient
from .errors import Provider
from .gemini import GeminiClient
from .grok import GrokClient
from .mcp import _PROTOCOL_VERSION, McpStdio, McpStreamable
from .mistral import MistralClient
from .models import ToolCall, ToolDefinition, UsageToken
from .openai import OpenAIClient

__all__ = ("instrument", "is_instrumented", "uninstrument")

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

# chat span of the in-flight complete_chat, for provider-specific enrichment
_active_chat_span: ContextVar[trace.Span | None] = ContextVar(
    "padwan_llm_otel_chat_span", default=None
)


@dataclass
class _AgentCounters:
    conversation_id: str
    inference_calls: int = 0
    tool_calls: int = 0


@dataclass
class _ActiveTool:
    span: trace.Span
    error_type: str | None = None
    error_message: str | None = None
    error_recorded: bool = False


_active_agent_counters: ContextVar[_AgentCounters | None] = ContextVar(
    "padwan_llm_otel_agent_counters", default=None
)
_active_tool: ContextVar[_ActiveTool | None] = ContextVar(
    "padwan_llm_otel_tool", default=None
)


@dataclass
class _Instruments:
    tracer: trace.Tracer
    logger: Logger
    duration: metrics.Histogram
    tokens: metrics.Histogram
    time_to_first_chunk: metrics.Histogram
    time_per_output_chunk: metrics.Histogram
    agent_duration: metrics.Histogram
    agent_inference_calls: metrics.Histogram
    agent_tool_calls: metrics.Histogram
    tool_duration: metrics.Histogram
    mcp_duration: metrics.Histogram
    mcp_session_duration: metrics.Histogram
    capture_content: bool


def instrument(
    tracer_provider: trace.TracerProvider | None = None,
    meter_provider: metrics.MeterProvider | None = None,
    logger_provider: LoggerProvider | None = None,
    capture_content: bool = False,
) -> None:
    """Emit opt-in OTel GenAI telemetry for all provider clients."""
    if _originals:
        return
    tracer = trace.get_tracer("padwan_llm", __version__, tracer_provider)
    meter = metrics.get_meter("padwan_llm", __version__, meter_provider)
    inst = _Instruments(
        tracer=tracer,
        logger=get_logger("padwan_llm", __version__, logger_provider),
        duration=meter.create_histogram(
            "gen_ai.client.operation.duration",
            unit="s",
            description="GenAI operation duration",
        ),
        tokens=meter.create_histogram(
            "gen_ai.client.token.usage",
            unit="{token}",
            description="Number of input and output tokens used",
        ),
        time_to_first_chunk=meter.create_histogram(
            "gen_ai.client.operation.time_to_first_chunk",
            unit="s",
            description="Time to receive the first streamed chunk",
        ),
        time_per_output_chunk=meter.create_histogram(
            "gen_ai.client.operation.time_per_output_chunk",
            unit="s",
            description="Time between consecutive streamed chunks",
        ),
        agent_duration=meter.create_histogram(
            "gen_ai.invoke_agent.duration",
            unit="s",
            description="Duration of an agent invocation",
        ),
        agent_inference_calls=meter.create_histogram(
            "gen_ai.invoke_agent.inference_calls",
            unit="{inference_call}",
            description="Inference calls made by an agent invocation",
        ),
        agent_tool_calls=meter.create_histogram(
            "gen_ai.invoke_agent.tool_calls",
            unit="{tool_call}",
            description="Tool calls made by an agent invocation",
        ),
        tool_duration=meter.create_histogram(
            "gen_ai.execute_tool.duration",
            unit="s",
            description="Duration of a tool execution",
        ),
        mcp_duration=meter.create_histogram(
            "mcp.client.operation.duration",
            unit="s",
            description="Duration of MCP client operations",
        ),
        mcp_session_duration=meter.create_histogram(
            "mcp.client.session.duration",
            unit="s",
            description="Duration of MCP client sessions",
        ),
        capture_content=capture_content,
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

    # OpenAI vendor extras live only on raw request and response payloads.
    _originals[OpenAIClient]["complete"] = OpenAIClient.complete
    _originals[OpenAIClient]["stream"] = OpenAIClient.stream
    setattr(OpenAIClient, "complete", _wrap_openai_complete(OpenAIClient.complete))
    setattr(OpenAIClient, "stream", _wrap_openai_stream(OpenAIClient.stream))

    _originals[AgentSession] = {
        "_dispatch_one": AgentSession._dispatch_one,
        "stream": AgentSession.stream,
    }
    setattr(
        AgentSession,
        "_dispatch_one",
        _wrap_execute_tool(AgentSession._dispatch_one, inst),
    )
    setattr(AgentSession, "stream", _wrap_invoke_agent(AgentSession.stream, inst))

    for mcp_cls in (McpStreamable, McpStdio):
        methods = {
            "__aenter__": mcp_cls.__aenter__,
            "__aexit__": mcp_cls.__aexit__,
            "_initialize": mcp_cls._initialize,
            "_refresh_tools": mcp_cls._refresh_tools,
            "_call": mcp_cls._call,
            "ping": mcp_cls.ping,
        }
        _originals[mcp_cls] = methods
        setattr(mcp_cls, "__aenter__", _wrap_mcp_enter(methods["__aenter__"], inst))
        setattr(mcp_cls, "__aexit__", _wrap_mcp_exit(methods["__aexit__"], inst))
        for name, method in (
            ("_initialize", "initialize"),
            ("_refresh_tools", "tools/list"),
            ("ping", "ping"),
        ):
            setattr(
                mcp_cls,
                name,
                _wrap_mcp_operation(methods[name], inst, method),
            )
        setattr(
            mcp_cls,
            "_call",
            _wrap_mcp_operation(methods["_call"], inst, "tools/call"),
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


def is_instrumented() -> bool:
    """Return whether Padwan OpenTelemetry instrumentation is active."""
    return bool(_originals)


def uninstrument() -> None:
    """Restore the original client methods."""
    for cls, methods in _originals.items():
        for name, fn in methods.items():
            setattr(cls, name, fn)
    _originals.clear()


def _request_attrs(
    client: LLMClientBase | RealtimeClientBase[Any], op: str = "chat"
) -> dict[str, Any]:
    """Return low-cardinality attributes shared by spans and metrics."""
    attrs: dict[str, Any] = {"gen_ai.operation.name": op}
    if provider := getattr(client, "provider", None):
        attrs["gen_ai.provider.name"] = _PROVIDER_NAMES.get(provider, provider)
    if model := getattr(client, "model", None):
        attrs["gen_ai.request.model"] = model
    if counters := _active_agent_counters.get():
        attrs["gen_ai.conversation.id"] = counters.conversation_id
    endpoint = urlsplit(getattr(client, "base_url", ""))
    if endpoint.hostname:
        attrs["server.address"] = endpoint.hostname
        port = endpoint.port or {
            "http": 80,
            "https": 443,
            "ws": 80,
            "wss": 443,
        }.get(endpoint.scheme)
        if port is not None:
            attrs["server.port"] = port
    return attrs


def _content_parts(content: Any) -> list[dict[str, Any]]:
    """Convert content to semconv parts without binary payloads."""
    if isinstance(content, str):
        return [{"type": "text", "content": content}]
    parts: list[dict[str, Any]] = []
    for part in content or ():
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append({"type": "text", "content": part.get("text")})
        elif isinstance(part, dict):
            parts.append({"type": part.get("type", "unknown")})
    return parts


def _capture_input(messages: Sequence[Any], *, separate_system: bool) -> dict[str, str]:
    """Serialize input messages as semconv span attributes."""
    system: list[dict[str, Any]] = []
    msgs: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system" and separate_system:
            system.append({"type": "text", "content": m.get("content")})
            continue
        if role == "tool":
            msgs.append(
                {
                    "role": "tool",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": m.get("tool_call_id"),
                            "response": m.get("content"),
                        }
                    ],
                }
            )
            continue
        parts = _content_parts(m.get("content"))
        for tc in m.get("tool_calls") or ():
            parts.append(
                {
                    "type": "tool_call",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                }
            )
        msgs.append({"role": role, "parts": parts})
    attrs = {"gen_ai.input.messages": _json_dumps(msgs)}
    if system:
        attrs["gen_ai.system_instructions"] = _json_dumps(system)
    return attrs


def _capture_tools(tools: Sequence[ToolDefinition] | None) -> str | None:
    if not tools:
        return None
    return _json_dumps([{"type": "function", **tool} for tool in tools])


def _structured_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    structured = dict(attrs)
    for name in (
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
    ):
        if value := structured.get(name):
            structured[name] = _json_loads(value)
    return structured


def _emit_inference_details(
    inst: _Instruments, span: trace.Span, attrs: dict[str, Any]
) -> None:
    inst.logger.emit(
        event_name="gen_ai.client.inference.operation.details",
        context=trace.set_span_in_context(span),
        attributes=_structured_attrs(attrs),
    )


def _capture_output(
    content: str | None,
    tool_calls: Sequence[ToolCall] | None,
    finish_reason: str | None,
) -> str:
    parts: list[dict[str, Any]] = []
    if content:
        parts.append({"type": "text", "content": content})
    for tc in tool_calls or ():
        parts.append(
            {
                "type": "tool_call",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "arguments": tc["function"]["arguments"],
            }
        )
    msg: dict[str, Any] = {"role": "assistant", "parts": parts}
    if finish_reason:
        msg["finish_reason"] = finish_reason
    return _json_dumps([msg])


def _usage_attrs(usage: UsageToken | None) -> dict[str, int]:
    if usage is None:
        return {}
    attrs = {
        "gen_ai.usage.input_tokens": usage["input"],
        "gen_ai.usage.output_tokens": usage["output"],
    }
    for key, attr in (
        ("reasoning", "gen_ai.usage.reasoning.output_tokens"),
        ("cached", "gen_ai.usage.cache_read.input_tokens"),
        ("cache_write", "gen_ai.usage.cache_write.input_tokens"),
    ):
        if (value := usage.get(key)) is not None:
            attrs[attr] = value
    return attrs


def _set_span_error(
    span: trace.Span,
    error_type: str,
    message: str,
    error: BaseException | None = None,
) -> None:
    span.set_attribute("error.type", error_type)
    span.set_status(StatusCode.ERROR, message)
    if error is not None:
        span.record_exception(error)


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
        _set_span_error(span, error_type, str(error), error)
        inst.logger.emit(
            event_name="gen_ai.client.operation.exception",
            severity_number=SeverityNumber.WARN,
            context=trace.set_span_in_context(span),
            attributes=metric_attrs,
            exception=error,
        )
    inst.duration.record(elapsed, metric_attrs)
    if usage is not None:
        for key, value in _usage_attrs(usage).items():
            span.set_attribute(key, value)
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


def _chat_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Sequence[Any]:
    return kwargs.get("messages") or (args[0] if args else ())


def _chat_tools(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Sequence[ToolDefinition] | None:
    if "tools" in kwargs:
        return kwargs["tools"]
    return args[1] if len(args) > 1 else None


def _capture_request(
    client: LLMClientBase, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, str]:
    attrs = _capture_input(
        _chat_messages(args, kwargs),
        separate_system=client.provider in {"anthropic", "gemini"},
    )
    if tools := _capture_tools(_chat_tools(args, kwargs)):
        attrs["gen_ai.tool.definitions"] = tools
    return attrs


def _wrap_complete_chat(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(self: LLMClientBase, *args: Any, **kwargs: Any) -> Any:
        attrs = _request_attrs(self)
        span = _start_span(inst, attrs, _sent_temperature(self))
        content_attrs = (
            _capture_request(self, args, kwargs) if inst.capture_content else {}
        )
        if inst.capture_content:
            for key, value in content_attrs.items():
                span.set_attribute(key, value)
        start = time.perf_counter()
        token = _active_chat_span.set(span)
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
                if inst.capture_content:
                    _emit_inference_details(
                        inst,
                        span,
                        {**attrs, **content_attrs, "error.type": type(e).__qualname__},
                    )
                _record_end(inst, span, attrs, start, error=e)
                raise
            finally:
                _active_chat_span.reset(token)
        if inst.capture_content:
            output = _capture_output(
                response.get("content"),
                response.get("tool_calls"),
                response.get("finish_reason"),
            )
            span.set_attribute("gen_ai.output.messages", output)
            detail_attrs: dict[str, Any] = {
                **attrs,
                **content_attrs,
                **_usage_attrs(usage),
                "gen_ai.output.messages": output,
                "gen_ai.response.finish_reasons": [response["finish_reason"]],
            }
            if (temperature := _sent_temperature(self)) is not None:
                detail_attrs["gen_ai.request.temperature"] = temperature
            _emit_inference_details(inst, span, detail_attrs)
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
        input_attrs = (
            _capture_request(self, args, kwargs) if inst.capture_content else None
        )
        attrs = _request_attrs(self)
        attrs["gen_ai.request.stream"] = True
        return _InstrumentedChatStream(
            stream,
            inst,
            attrs,
            _sent_temperature(self),
            thinking,
            input_attrs,
        )

    return wrapper


def _set_openai_request_attrs(span: trace.Span, body: Any) -> None:
    span.set_attribute("openai.api.type", "chat_completions")
    if tier := body.get("service_tier"):
        span.set_attribute("openai.request.service_tier", tier)


def _set_openai_response_attrs(span: trace.Span, data: Any) -> None:
    if not isinstance(data, dict):
        return
    if tier := data.get("service_tier"):
        span.set_attribute("openai.response.service_tier", tier)
    if fingerprint := data.get("system_fingerprint"):
        span.set_attribute("openai.response.system_fingerprint", fingerprint)


def _wrap_openai_complete(original: Any) -> Any:
    """Capture OpenAI attributes from raw non-streaming payloads."""

    @functools.wraps(original)
    async def wrapper(self: Any, body: Any, *args: Any, **kwargs: Any) -> Any:
        span = _active_chat_span.get()
        if span is not None and span.is_recording():
            _set_openai_request_attrs(span, body)
        result = await original(self, body, *args, **kwargs)
        if span is not None and span.is_recording():
            _set_openai_response_attrs(span, result[0])
        return result

    return wrapper


def _wrap_openai_stream(original: Any) -> Any:
    """Capture OpenAI attributes from raw streaming payloads."""

    @functools.wraps(original)
    async def wrapper(
        self: Any, body: Any, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        span = _active_chat_span.get()
        if span is not None and span.is_recording():
            _set_openai_request_attrs(span, body)
        async for chunk in original(self, body, *args, **kwargs):
            if span is not None and span.is_recording():
                _set_openai_response_attrs(span, chunk)
            yield chunk

    return wrapper


def _wrap_invoke_agent(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(
        self: AgentSession, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str]:
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": "invoke_agent",
            "gen_ai.conversation.id": self.session_id,
        }
        if model := getattr(self.client, "model", None):
            attrs["gen_ai.request.model"] = model
        span = inst.tracer.start_span(
            "invoke_agent", kind=SpanKind.INTERNAL, attributes=attrs
        )
        ctx = trace.set_span_in_context(span)
        before = cast(dict[str, int], dict(self.total_usage))
        before_messages = len(self.messages)
        counters = _AgentCounters(conversation_id=self.session_id)
        start = time.perf_counter()
        error: BaseException | None = None
        it = original(self, *args, **kwargs).__aiter__()
        try:
            while True:
                context_token = otel_context.attach(ctx)
                counter_token = _active_agent_counters.set(counters)
                try:
                    chunk = await anext(it)
                except StopAsyncIteration:
                    break
                finally:
                    _active_agent_counters.reset(counter_token)
                    otel_context.detach(context_token)
                yield chunk
        except GeneratorExit:
            raise
        except BaseException as e:
            error = e
            raise
        finally:
            after = cast(dict[str, int], dict(self.total_usage))
            new_messages = self.messages[before_messages:]
            counters.inference_calls = max(
                counters.inference_calls,
                sum(message["role"] == "assistant" for message in new_messages),
            )
            counters.tool_calls = max(
                counters.tool_calls,
                sum(message["role"] == "tool" for message in new_messages),
            )
            usage: UsageToken = {
                "total": after.get("total", 0) - before.get("total", 0),
                "input": after.get("input", 0) - before.get("input", 0),
                "output": after.get("output", 0) - before.get("output", 0),
            }
            if "cached" in after or "cached" in before:
                usage["cached"] = after.get("cached", 0) - before.get("cached", 0)
            if "reasoning" in after or "reasoning" in before:
                usage["reasoning"] = after.get("reasoning", 0) - before.get(
                    "reasoning", 0
                )
            if "cache_write" in after or "cache_write" in before:
                usage["cache_write"] = after.get("cache_write", 0) - before.get(
                    "cache_write", 0
                )
            for key, value in _usage_attrs(usage).items():
                span.set_attribute(key, value)
            metric_attrs = {
                key: value
                for key, value in attrs.items()
                if key == "gen_ai.request.model"
            }
            if error is not None:
                error_type = type(error).__qualname__
                metric_attrs["error.type"] = error_type
                _set_span_error(span, error_type, str(error), error)
            elapsed = time.perf_counter() - start
            inst.agent_duration.record(elapsed, metric_attrs)
            inst.agent_inference_calls.record(counters.inference_calls)
            inst.agent_tool_calls.record(counters.tool_calls)
            span.end()

    return wrapper


def _mcp_attrs(client: McpStreamable | McpStdio) -> dict[str, Any]:
    attrs: dict[str, Any] = {"mcp.protocol.version": _PROTOCOL_VERSION}
    if isinstance(client, McpStdio):
        attrs["network.transport"] = "pipe"
        return attrs
    attrs["network.transport"] = "tcp"
    attrs["network.protocol.name"] = "http"
    endpoint = urlsplit(client.url)
    if endpoint.hostname:
        attrs["server.address"] = endpoint.hostname
        port = endpoint.port or {"http": 80, "https": 443}.get(endpoint.scheme)
        if port is not None:
            attrs["server.port"] = port
    if client._session_id:
        attrs["mcp.session.id"] = client._session_id
    return attrs


def _mcp_error(error: BaseException) -> tuple[str, str | None]:
    message = str(error)
    if message.startswith("MCP error ") and ":" in message:
        status = message.removeprefix("MCP error ").split(":", 1)[0]
        return status, status
    return type(error).__qualname__, None


def _wrap_mcp_operation(original: Any, inst: _Instruments, method: str) -> Any:
    @functools.wraps(original)
    async def wrapper(self: McpStreamable | McpStdio, *args: Any, **kwargs: Any) -> Any:
        attrs = {**_mcp_attrs(self), "mcp.method.name": method}
        name = args[0] if method == "tools/call" and args else kwargs.get("name")
        arguments = (
            args[1] if method == "tools/call" and len(args) > 1 else kwargs.get("args")
        )
        if name:
            attrs["gen_ai.operation.name"] = "execute_tool"
            attrs["gen_ai.tool.name"] = name
        active = _active_tool.get() if method == "tools/call" else None
        owns_span = active is None
        span = (
            inst.tracer.start_span(
                f"{method} {name}" if name else method,
                kind=SpanKind.CLIENT,
                attributes=attrs,
            )
            if active is None
            else active.span
        )
        if active is not None:
            for key, value in attrs.items():
                if key != "gen_ai.tool.name":
                    span.set_attribute(key, value)
        if inst.capture_content and arguments is not None:
            span.set_attribute("gen_ai.tool.call.arguments", _json_dumps(arguments))
        metric_attrs = dict(attrs)
        start = time.perf_counter()
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                result = await original(self, *args, **kwargs)
                if isinstance(result, dict) and result.get("isError") is True:
                    metric_attrs["error.type"] = "tool_error"
                    _set_span_error(span, "tool_error", "MCP tool returned an error")
                    if active is not None:
                        active.error_type = "tool_error"
                        active.error_message = "MCP tool returned an error"
                        active.error_recorded = True
                if inst.capture_content and method == "tools/call":
                    span.set_attribute("gen_ai.tool.call.result", _json_dumps(result))
                return result
            except BaseException as e:
                error_type, status = _mcp_error(e)
                metric_attrs["error.type"] = error_type
                if status is not None:
                    metric_attrs["rpc.response.status_code"] = status
                    span.set_attribute("rpc.response.status_code", status)
                _set_span_error(span, error_type, str(e), e)
                if active is not None:
                    active.error_type = error_type
                    active.error_message = str(e)
                    active.error_recorded = True
                raise
            finally:
                if owns_span:
                    span.end()
                inst.mcp_duration.record(time.perf_counter() - start, metric_attrs)

    return wrapper


def _wrap_mcp_enter(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(self: McpStreamable | McpStdio) -> Any:
        start = time.perf_counter()
        try:
            result = await original(self)
        except BaseException as e:
            attrs = _mcp_attrs(self)
            attrs["error.type"] = type(e).__qualname__
            inst.mcp_session_duration.record(time.perf_counter() - start, attrs)
            raise
        setattr(self, "_otel_mcp_session_start", start)
        return result

    return wrapper


def _wrap_mcp_exit(original: Any, inst: _Instruments) -> Any:
    @functools.wraps(original)
    async def wrapper(self: McpStreamable | McpStdio, *args: Any) -> None:
        start = getattr(self, "_otel_mcp_session_start", None)
        attrs = _mcp_attrs(self)
        error = args[1] if len(args) > 1 else None
        try:
            await original(self, *args)
        except BaseException as e:
            error = e
            raise
        finally:
            setattr(self, "_otel_mcp_session_start", None)
            if start is not None:
                if error is not None:
                    attrs["error.type"] = type(error).__qualname__
                inst.mcp_session_duration.record(time.perf_counter() - start, attrs)

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
        if counters := _active_agent_counters.get():
            counters.tool_calls += 1
        span_attrs: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": name,
            "gen_ai.tool.type": "function",
            "gen_ai.tool.call.id": tc["id"],
        }
        if counters := _active_agent_counters.get():
            span_attrs["gen_ai.conversation.id"] = counters.conversation_id
        tool = args[0] if args else None
        call_args = args[1] if len(args) > 1 else None
        approved = args[2] if len(args) > 2 else True
        if tool is not None:
            span_attrs["gen_ai.tool.description"] = tool.description
        span = inst.tracer.start_span(
            f"execute_tool {name}", kind=SpanKind.INTERNAL, attributes=span_attrs
        )
        if inst.capture_content and call_args is not None:
            span.set_attribute("gen_ai.tool.call.arguments", _json_dumps(call_args))
        active = _ActiveTool(span)
        if tool is None:
            active.error_type = "unknown_tool"
        elif not approved:
            active.error_type = "tool_denied"
        metric_attrs: dict[str, Any] = {
            "gen_ai.tool.name": name,
            "gen_ai.tool.type": "function",
        }
        start = time.perf_counter()
        active_token = _active_tool.set(active)
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                result = await original(self, tc, *args, **kwargs)
                if active.error_type is None:
                    try:
                        parsed = _json_loads(result)
                    except (TypeError, ValueError):
                        parsed = None
                    if isinstance(parsed, dict) and "error" in parsed:
                        active.error_type = "tool_error"
                        active.error_message = "Tool returned an error"
                if inst.capture_content:
                    span.set_attribute("gen_ai.tool.call.result", _json_dumps(result))
                return result
            except BaseException as e:
                active.error_type = type(e).__qualname__
                active.error_message = str(e)
                _set_span_error(span, active.error_type, str(e), e)
                active.error_recorded = True
                raise
            finally:
                _active_tool.reset(active_token)
                if active.error_type is not None:
                    metric_attrs["error.type"] = active.error_type
                    if not active.error_recorded:
                        _set_span_error(
                            span,
                            active.error_type,
                            active.error_message or active.error_type,
                        )
                span.end()
                inst.tool_duration.record(time.perf_counter() - start, metric_attrs)

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
    """Delegate to a provider stream while recording its lifecycle."""

    def __init__(
        self,
        inner: ChatStream,
        inst: _Instruments,
        attrs: dict[str, Any],
        temperature: float | None,
        thinking: _ThoughtTimer | None = None,
        input_attrs: dict[str, str] | None = None,
    ) -> None:
        self._inner = inner
        self._inst = inst
        self._attrs = attrs
        self._temperature = temperature
        self._thinking = thinking
        self._input_attrs = input_attrs

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":  # guard against recursion before __init__ ran
            raise AttributeError(name)
        return getattr(self._inner, name)

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        span = _start_span(self._inst, self._attrs, self._temperature)
        if counters := _active_agent_counters.get():
            counters.inference_calls += 1
        if self._input_attrs:
            for key, value in self._input_attrs.items():
                span.set_attribute(key, value)
        start = time.perf_counter()
        first: float | None = None
        previous = start
        chunks: list[str] = []
        error: BaseException | None = None
        chat_token = _active_chat_span.set(span)
        with trace.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ):
            try:
                async for chunk in self._inner:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                        span.set_attribute(
                            "gen_ai.response.time_to_first_chunk", first - start
                        )
                        self._inst.time_to_first_chunk.record(
                            first - start, self._attrs
                        )
                    else:
                        self._inst.time_per_output_chunk.record(
                            now - previous, self._attrs
                        )
                    previous = now
                    if self._input_attrs is not None:
                        chunks.append(chunk)
                    yield chunk
            except GeneratorExit:
                raise
            except BaseException as e:
                error = e
                raise
            finally:
                self.usage = self._inner.usage
                self.tool_calls = self._inner.tool_calls
                reason = getattr(self._inner, "finish_reason", None)
                if self._input_attrs is not None:
                    output = _capture_output(
                        "".join(chunks) or None, self._inner.tool_calls, reason
                    )
                    span.set_attribute("gen_ai.output.messages", output)
                    detail_attrs: dict[str, Any] = {
                        **self._attrs,
                        **self._input_attrs,
                        **_usage_attrs(self._inner.usage),
                        "gen_ai.output.messages": output,
                    }
                    if self._temperature is not None:
                        detail_attrs["gen_ai.request.temperature"] = self._temperature
                    if reason:
                        detail_attrs["gen_ai.response.finish_reasons"] = [reason]
                    if error is not None:
                        detail_attrs["error.type"] = type(error).__qualname__
                    _emit_inference_details(self._inst, span, detail_attrs)
                _record_end(
                    self._inst,
                    span,
                    self._attrs,
                    start,
                    usage=self._inner.usage,
                    finish_reasons=[reason] if reason else None,
                    tool_names=_tool_names(self._inner.tool_calls),
                    thinking=self._thinking,
                    error=error,
                )
                _active_chat_span.reset(chat_token)
