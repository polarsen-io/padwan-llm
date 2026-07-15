from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict

__all__ = (
    "AnthropicContentBlock",
    "AnthropicMessage",
    "AnthropicTool",
    "AnthropicUsage",
    "MessagesBody",
    "MessagesResponse",
    "StopReason",
)


class AnthropicTool(TypedDict):
    """A tool definition in Anthropic Messages API format."""

    name: str
    description: str
    input_schema: dict[str, Any]


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
    """Request body for POST /v1/messages."""

    model: str
    max_tokens: int
    messages: list[AnthropicMessage]
    system: NotRequired[str]
    tools: NotRequired[list[AnthropicTool]]
    stream: NotRequired[bool]


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
    usage: AnthropicUsage
