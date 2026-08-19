from __future__ import annotations

import dataclasses
import typing
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import field
from functools import partial
from typing import ClassVar, Literal, cast, get_args

import niquests
from urllib3.util.retry import Retry

from .._base import ChatStream, LLMClientBase, env_api_key
from ..conversation import AssistantToolMessage, ChatMessage, ToolResultMessage
from ..errors import LLMError, Provider, TooManyRequestsError
from ..models import (
    ChatResponse,
    FinishReason,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    UsageToken,
)
from .models import (
    AnthropicMessage,
    MessagesBody,
    MessagesResponse,
    StopReason,
)
from .tools import AnthropicToolMixin

__all__ = (
    "ANTHROPIC_ENDPOINT",
    "ANTHROPIC_MODELS",
    "AnthropicChatStream",
    "AnthropicClient",
    "AnthropicModel",
    "is_anthropic_model",
)

AnthropicModel = Literal[
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-opus-4-1",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
]

ANTHROPIC_MODELS: set[str] = set(get_args(AnthropicModel))

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/"

ANTHROPIC_VERSION = "2023-06-01"


def is_anthropic_model(model_name: str | None) -> bool:
    """Check if the model name is an Anthropic model."""
    if model_name is None:
        return False
    return model_name.startswith("claude")


_STOP_REASON_MAP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


def _check_resp_status(resp: niquests.Response) -> niquests.Response:
    """Check HTTP status and raise appropriate errors without consuming the body."""
    try:
        return resp.raise_for_status()
    except niquests.exceptions.HTTPError as e:
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("retry-after", "60"))
            raise TooManyRequestsError(retry_delay=retry_after, response=resp) from e
        try:
            data = resp.json()
        except Exception:
            raise e
        message = (data or {}).get("error", {}).get("message", "")
        raise LLMError("anthropic", f"{resp.status_code} {message}", body=data) from e


def _check_resp[T](
    resp: niquests.Response, *, decoder: Callable[[bytes], T] | None = None
) -> T:
    """Check Anthropic HTTP response and return the parsed JSON body,
    deserialized through `decoder` when provided (e.g. msgspec for typed validation)."""
    checked = _check_resp_status(resp)
    if decoder is None:
        return checked.json()
    if not (body := checked.content):
        raise LLMError("anthropic", "Empty response body")
    return decoder(body)


def _usage_from_anthropic(usage: dict[str, typing.Any] | None) -> UsageToken:
    """Map an Anthropic usage object to a UsageToken."""
    usage = usage or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    token: UsageToken = {
        "total": input_tokens + output_tokens,
        "input": input_tokens,
        "output": output_tokens,
    }
    if (cached := usage.get("cache_read_input_tokens")) is not None:
        token["cached"] = cached
    return token


@dataclasses.dataclass
class AnthropicClient(LLMClientBase[Retry], AnthropicToolMixin):
    """Anthropic Messages API client (chat + tool use + streaming).

    Sampling parameters are rejected on current Claude models, so the inherited
    `temperature` field is never sent — steer behavior via the prompt instead.
    """

    provider: ClassVar[Provider] = "anthropic"

    def _get_default_api_key(self) -> str:
        return env_api_key(self.provider, "ANTHROPIC_API_KEY")

    _deprecations: ClassVar[Mapping[str, str]] = {"claude-opus-4-1": "2026-08-05"}
    model: str | None = "claude-opus-4-8"
    base_url: str = ANTHROPIC_ENDPOINT
    max_tokens: int = 4096
    """Maximum output tokens per response (required by the Messages API)."""
    _retry: Retry = field(
        default_factory=partial(
            Retry,
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504, 529],
            allowed_methods=["POST"],
        )
    )

    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        session.headers["x-api-key"] = self._api_key
        session.headers["anthropic-version"] = ANTHROPIC_VERSION

    async def complete(self, body: MessagesBody) -> tuple[MessagesResponse, UsageToken]:
        """Fetch a completion from the Messages API."""
        resp = await self.session.post("/messages", json=body)
        data: MessagesResponse = _check_resp(resp)
        return data, _usage_from_anthropic(cast(dict, data.get("usage")))

    async def stream(self, body: MessagesBody) -> AsyncIterator[dict]:
        """Stream a Messages API request, yielding parsed SSE events as dicts."""
        resp = await self.session.post(
            self._sse_url("/messages"), json={**body, "stream": True}
        )
        _check_resp_status(resp)
        ext = resp.extension
        if ext is None:
            raise LLMError(self.provider, "SSE extension not available on response")
        while not ext.closed:
            event = await ext.next_payload()
            if event is None:
                break
            if not event.data:
                continue
            try:
                payload = event.json()
            except ValueError as e:
                raise LLMError(self.provider, f"Stream parse error: {e}") from e
            if payload.get("type") == "error":
                error = payload.get("error", {})
                raise LLMError(
                    self.provider, error.get("message", "stream error"), body=payload
                )
            yield payload

    def build_body(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> MessagesBody:
        """Build a Messages API body: system messages become the system field."""
        if not self.model:
            raise LLMError(self.provider, "No model specified")
        converted: list[AnthropicMessage] = []
        system_parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_parts.append(cast(str, msg["content"]))
                continue
            if role == "tool":
                converted.append(
                    self._convert_tool_result(cast(ToolResultMessage, msg))
                )
                continue
            if role == "assistant" and "tool_calls" in msg:
                converted.append(
                    self._convert_assistant_tool_message(
                        cast(AssistantToolMessage, msg)
                    )
                )
                continue
            converted.append(
                {
                    "role": cast(Literal["user", "assistant"], role),
                    "content": cast(str, msg["content"]),
                }
            )
        body: MessagesBody = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if tools:
            body["tools"] = self._tools_to_anthropic(tools)
        return body

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> tuple[ChatResponse, UsageToken]:
        """Send a chat conversation and return the structured response."""
        data, token = await self.complete(self.build_body(messages, tools))
        blocks = data.get("content") or []

        # Thinking blocks are forwarded to `on_thought` (when summarized text is
        # present) and never returned as answer text.
        text: str | None = None
        for block in blocks:
            block_type = block.get("type")
            if block_type == "thinking":
                if self.on_thought and (thought := block.get("thinking")):
                    self.on_thought(thought)
                continue
            if block_type == "text" and text is None:
                text = block.get("text")

        tool_calls = self._extract_anthropic_tool_calls(blocks)
        raw_reason = cast("StopReason | None", data.get("stop_reason"))
        if tool_calls:
            finish_reason: FinishReason = "tool_calls"
        else:
            finish_reason = _STOP_REASON_MAP.get(raw_reason or "", "other")

        response: ChatResponse = {"content": text, "finish_reason": finish_reason}
        if tool_calls:
            response["tool_calls"] = tool_calls
        if text is None and not tool_calls:
            if raw_reason == "refusal":
                raise LLMError(self.provider, "Request refused by safety classifiers")
            raise LLMError(self.provider, "No text content in response")
        return response, token

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
        extra_params: dict[str, typing.Any] | None = None,
    ) -> AnthropicChatStream:
        """Stream a chat conversation, yielding text chunks.

        Usage and tool_calls are available on the returned stream object after
        iteration.
        """
        return AnthropicChatStream(
            self, messages, tools, extra_params, on_thought=self.on_thought
        )


class AnthropicChatStream(ChatStream, AnthropicToolMixin):
    """ChatStream implementation for the Anthropic Messages API."""

    def __init__(
        self,
        client: AnthropicClient,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
        extra_params: dict[str, typing.Any] | None = None,
        on_thought: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._messages = messages
        self._tools = tools
        self._extra_params = extra_params
        self._on_thought = on_thought
        self.usage: UsageToken | None = None
        self.tool_calls: list[ToolCall] | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        body = self._client.build_body(self._messages, self._tools)
        if self._extra_params:
            body = cast(MessagesBody, {**body, **self._extra_params})

        input_tokens = 0
        output_tokens = 0
        cached: int | None = None
        # index -> partially assembled tool call; input arrives as input_json_delta
        pending: dict[int, tuple[str, str, list[str]]] = {}

        async for event in self._client.stream(body):
            match event.get("type"):
                case "message_start":
                    usage = event.get("message", {}).get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    cached = usage.get("cache_read_input_tokens")
                case "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        pending[event.get("index", 0)] = (
                            block.get("id", ""),
                            block.get("name", ""),
                            [],
                        )
                case "content_block_delta":
                    delta = event.get("delta", {})
                    match delta.get("type"):
                        case "text_delta":
                            yield delta.get("text", "")
                        case "input_json_delta":
                            if (
                                entry := pending.get(event.get("index", 0))
                            ) is not None:
                                entry[2].append(delta.get("partial_json", ""))
                        case "thinking_delta":
                            if self._on_thought and (thought := delta.get("thinking")):
                                self._on_thought(thought)
                case "message_delta":
                    output_tokens = event.get("usage", {}).get(
                        "output_tokens", output_tokens
                    )

        token: UsageToken = {
            "total": input_tokens + output_tokens,
            "input": input_tokens,
            "output": output_tokens,
        }
        if cached is not None:
            token["cached"] = cached
        self.usage = token

        if pending:
            self.tool_calls = [
                ToolCall(
                    id=call_id,
                    type="function",
                    function=ToolCallFunction(
                        name=name, arguments="".join(parts) or "{}"
                    ),
                )
                for _, (call_id, name, parts) in sorted(pending.items())
            ]
