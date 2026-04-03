from typing import Any, Literal, NotRequired, TypedDict

__all__ = (
    "ChatResponse",
    "FinishReason",
    "ToolCall",
    "ToolCallFunction",
    "ToolDefinition",
    "UsageToken",
)


class UsageToken(TypedDict):
    """Token usage information for LLM interactions."""

    total: int
    input: int
    output: int
    cached: NotRequired[int]


# Tool calling


class ToolCallFunction(TypedDict):
    """Function details within a tool call."""

    name: str
    arguments: str


class ToolCall(TypedDict):
    """A tool invocation requested by the model."""

    id: str
    type: Literal["function"]
    function: ToolCallFunction
    thought_signature: NotRequired[str]


class ToolDefinition(TypedDict):
    """A tool the model may call, described by a JSON Schema for its parameters."""

    name: str
    description: str
    parameters: dict[str, Any]


FinishReason = Literal[
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "error",
    "other",
]


class ChatResponse(TypedDict):
    """Structured response from a chat completion, supporting both text and tool calls."""

    content: str | None
    tool_calls: NotRequired[list[ToolCall]]
    finish_reason: FinishReason
