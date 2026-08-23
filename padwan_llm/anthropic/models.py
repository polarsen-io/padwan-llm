from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

__all__ = (
    "AnthropicContentBlock",
    "AnthropicMessage",
    "AnthropicTool",
    "AnthropicToolChoice",
    "AnthropicUsage",
    "CountTokensResponse",
    "MessagesBody",
    "MessagesResponse",
    "StopReason",
)


class AnthropicTool(TypedDict):
    """A tool definition in Anthropic Messages API format.

    Server tools (web search, code execution, ...) arrive with an
    API-versioned `type` instead of an `input_schema`.
    """

    name: str
    description: NotRequired[str]
    input_schema: NotRequired[dict[str, Any]]
    type: NotRequired[str]
    cache_control: NotRequired[dict[str, Any]]


class AnthropicToolChoice(TypedDict):
    """The `tool_choice` field of a Messages API request."""

    type: Literal["auto", "any", "tool", "none"]
    name: NotRequired[str]
    disable_parallel_tool_use: NotRequired[bool]


class AnthropicContentBlock(TypedDict, total=False):
    """A content block in a message or response (text, tool_use, tool_result, image, thinking)."""

    type: str
    text: str
    id: str
    name: str
    input: dict[str, Any]
    tool_use_id: str
    content: Any
    source: dict[str, Any]
    thinking: str
    signature: str
    cache_control: dict[str, Any]


class AnthropicMessage(TypedDict):
    """A single message in Anthropic wire format."""

    role: Literal["user", "assistant"]
    content: str | list[AnthropicContentBlock]


StopReason = Literal[
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
    "model_context_window_exceeded",
]


class MessagesBody(TypedDict):
    """Request body for POST /v1/messages.

    Covers both what `AnthropicClient` sends and what Anthropic-API callers
    (e.g. Claude Code behind the compat proxy) may send inbound.
    """

    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    system: NotRequired[str | list[AnthropicContentBlock]]
    tools: NotRequired[list[AnthropicTool]]
    tool_choice: NotRequired[AnthropicToolChoice]
    stream: NotRequired[bool]
    stop_sequences: NotRequired[list[str]]
    temperature: NotRequired[float]
    top_p: NotRequired[float]
    top_k: NotRequired[int]
    metadata: NotRequired[dict[str, Any]]
    thinking: NotRequired[dict[str, Any]]


class AnthropicUsage(TypedDict, total=False):
    """Usage object returned on messages and message_start/message_delta events."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class MessagesResponse(TypedDict, total=False):
    """Response body of POST /v1/messages (non-streaming)."""

    id: str
    type: str
    role: str
    model: str
    content: list[AnthropicContentBlock]
    stop_reason: StopReason | None
    stop_sequence: str | None
    usage: AnthropicUsage


class CountTokensResponse(TypedDict):
    """Response body of POST /v1/messages/count_tokens."""

    input_tokens: int
