from __future__ import annotations

import typing
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from .._json import loads as _json_loads
from ..errors import LLMError, QuotaExceededError, TooManyRequestsError
from ..logs import log
from ..openai.client import _extract_text_payload, _extract_thought_payload
from .models import AnthropicContentBlock, AnthropicUsage, MessagesResponse, StopReason

if TYPE_CHECKING:
    from ..openai.types import (
        CompletionUsage,
        CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse,
    )

__all__ = (
    "error_to_anthropic",
    "response_to_anthropic",
    "stream_to_anthropic",
)

AnthropicEvent = tuple[str, dict[str, Any]]
"""A named Anthropic SSE event: (event name, payload including its `type`)."""

_FINISH_TO_STOP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def _new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def _usage_to_anthropic(usage: CompletionUsage | None) -> AnthropicUsage:
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0}
    token: AnthropicUsage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
    if details := usage.get("prompt_tokens_details"):
        if (cached := details.get("cached_tokens")) is not None:
            token["cache_read_input_tokens"] = cached
    return token


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = _json_loads(arguments)
    except (ValueError, TypeError):
        log.debug("anthropic events: unparseable tool arguments %r", arguments[:200])
        return {}
    return parsed if isinstance(parsed, dict) else {}


def response_to_anthropic(
    data: CreateChatCompletionResponse, *, model: str | None = None
) -> MessagesResponse:
    """Translate an OpenAI chat completion to an Anthropic Messages response.

    `model` sets the model name reported back (e.g. the name the Anthropic
    caller requested); defaults to the backend's reported model.
    """
    choice = data["choices"][0]
    message = typing.cast(dict[str, Any], choice["message"])

    blocks: list[AnthropicContentBlock] = []
    if thought := _extract_thought_payload(message):
        blocks.append({"type": "thinking", "thinking": thought})
    if text := _extract_text_payload(message):
        blocks.append({"type": "text", "text": text})
    tool_calls = message.get("tool_calls") or []
    for i, tc in enumerate(tool_calls):
        function = tc.get("function") or {}
        arguments = function.get("arguments", "")
        blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"call_{i}",
                "name": function.get("name", ""),
                "input": arguments
                if isinstance(arguments, dict)
                else _parse_tool_arguments(arguments),
            }
        )

    raw_reason = choice.get("finish_reason") or "stop"
    stop_reason = (
        "tool_use" if tool_calls else _FINISH_TO_STOP.get(raw_reason, "end_turn")
    )
    return {
        "id": data.get("id") or _new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model or data.get("model", ""),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _usage_to_anthropic(data.get("usage")),
    }


class _BlockEmitter:
    """Tracks the currently open content block and assigns Anthropic block indexes."""

    def __init__(self) -> None:
        self.next_index = 0
        self.open_index: int | None = None

    def open(self, content_block: AnthropicContentBlock) -> list[AnthropicEvent]:
        events = self.close()
        self.open_index = self.next_index
        self.next_index += 1
        events.append(
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": self.open_index,
                    "content_block": dict(content_block),
                },
            )
        )
        return events

    def delta(self, delta: dict[str, Any]) -> AnthropicEvent:
        return (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": self.open_index,
                "delta": delta,
            },
        )

    def close(self) -> list[AnthropicEvent]:
        if self.open_index is None:
            return []
        event: AnthropicEvent = (
            "content_block_stop",
            {"type": "content_block_stop", "index": self.open_index},
        )
        self.open_index = None
        return [event]


async def stream_to_anthropic(
    chunks: AsyncIterator[CreateChatCompletionStreamResponse],
    *,
    model: str,
) -> AsyncIterator[AnthropicEvent]:
    """Translate an OpenAI chat completion stream to Anthropic SSE events.

    Yields (event name, payload) pairs following the Messages API sequence:
    message_start, content_block_start/delta/stop per block, message_delta,
    message_stop. OpenAI reports usage only on the final chunk, so
    message_start carries zeroed usage and the real counts land in
    message_delta — Anthropic clients (incl. Claude Code) accept this.
    """
    yield (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": _new_message_id(),
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    emitter = _BlockEmitter()
    # "thinking" | "text" | openai tool index currently streaming into the open block
    open_kind: str | int | None = None
    finish_reason: str | None = None
    usage: CompletionUsage | None = None
    saw_tool_call = False

    async for chunk in chunks:
        if chunk_usage := chunk.get("usage"):
            usage = chunk_usage
        choices = chunk.get("choices")
        if not choices:
            continue
        if reason := choices[0].get("finish_reason"):
            finish_reason = reason
        delta = typing.cast(dict[str, Any], choices[0].get("delta") or {})

        if thought := _extract_thought_payload(delta):
            if open_kind != "thinking":
                open_kind = "thinking"
                for event in emitter.open({"type": "thinking", "thinking": ""}):
                    yield event
            yield emitter.delta({"type": "thinking_delta", "thinking": thought})

        if text := _extract_text_payload(delta):
            if open_kind != "text":
                open_kind = "text"
                for event in emitter.open({"type": "text", "text": ""}):
                    yield event
            yield emitter.delta({"type": "text_delta", "text": text})

        for tc_chunk in delta.get("tool_calls") or []:
            saw_tool_call = True
            idx = tc_chunk.get("index", 0)
            function = tc_chunk.get("function") or {}
            if open_kind != idx:
                open_kind = idx
                for event in emitter.open(
                    {
                        "type": "tool_use",
                        "id": tc_chunk.get("id") or f"call_{idx}",
                        "name": function.get("name", ""),
                        "input": {},
                    }
                ):
                    yield event
            if arguments := function.get("arguments"):
                yield emitter.delta(
                    {"type": "input_json_delta", "partial_json": arguments}
                )

    for event in emitter.close():
        yield event

    stop_reason = (
        "tool_use"
        if saw_tool_call
        else _FINISH_TO_STOP.get(finish_reason or "stop", "end_turn")
    )
    yield (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": dict(_usage_to_anthropic(usage)),
        },
    )
    yield ("message_stop", {"type": "message_stop"})


def error_to_anthropic(exc: BaseException) -> tuple[int, dict[str, Any]]:
    """Map a padwan-llm error to an Anthropic error status and body.

    Returns (HTTP status, body) in the `{"type": "error", "error": {...}}`
    shape Anthropic clients expect, so their retry logic keeps working.
    """
    match exc:
        case TooManyRequestsError():
            status, error_type = 429, "rate_limit_error"
            message = exc.message or f"rate limited, retry in {exc.retry_delay}s"
        case QuotaExceededError():
            status, error_type = 400, "invalid_request_error"
            message = f"quota exceeded: {exc.body}"
        case LLMError():
            status, error_type = 502, "api_error"
            message = str(exc)
        case _:
            status, error_type = 500, "api_error"
            message = str(exc) or type(exc).__name__
    return status, {
        "type": "error",
        "error": {"type": error_type, "message": message},
    }
