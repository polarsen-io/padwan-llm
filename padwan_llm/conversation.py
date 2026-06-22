from dataclasses import dataclass, field
from typing import Literal, Self, TypedDict, cast

from .content import ContentPart
from .models import ChatResponse, ToolCall, UsageToken

__all__ = (
    "AssistantToolMessage",
    "ChatMessage",
    "ConversationSnapshot",
    "ConversationState",
    "Message",
    "ToolResultMessage",
)


class Message(TypedDict):
    """A single message in a conversation.

    User messages may carry a list of multimodal content parts (text + images);
    system and assistant messages always use a plain string.
    """

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class AssistantToolMessage(TypedDict):
    """An assistant message that contains tool calls instead of (or alongside) text."""

    role: Literal["assistant"]
    content: str | None
    tool_calls: list[ToolCall]


class ToolResultMessage(TypedDict):
    """The result of a tool invocation, sent back to the model.

    The `name` field carries the function name, required by Gemini's functionResponse.
    OpenAI uses `tool_call_id` for correlation and ignores `name`.
    """

    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


ChatMessage = Message | AssistantToolMessage | ToolResultMessage


class ConversationSnapshot(TypedDict):
    """Serializable view of a `ConversationState`.

    Round-tripped through `ConversationState.snapshot` /
    `ConversationState.from_snapshot`. JSON-serializable as long as the
    message contents are.
    """

    system: str | None
    messages: list[ChatMessage]
    total_usage: UsageToken


@dataclass
class ConversationState:
    """Pure state container for conversation history and usage tracking.

    Manages message history including an optional system prompt, and tracks
    token usage across multiple turns. This class contains no client or
    provider-specific logic—it's purely state management.
    """

    system: str | None = None
    messages: list[ChatMessage] = field(default_factory=list)
    last_usage: UsageToken | None = field(default=None, init=False)
    total_usage: UsageToken = field(
        default_factory=lambda: {"total": 0, "input": 0, "output": 0}, init=False
    )

    def __post_init__(self) -> None:
        if self.system:
            self.messages.insert(0, Message(role="system", content=self.system))

    def add_user_message(self, content: str | list[ContentPart]) -> Message:
        """Add a user message to the conversation history.

        Accepts either plain text or a list of multimodal content parts.
        """
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_response(self, response: ChatResponse) -> ChatMessage:
        """Add an assistant message from a ChatResponse.

        If the response contains tool calls, stores an AssistantToolMessage;
        otherwise stores a plain Message with the text content.
        """
        if tool_calls := response.get("tool_calls"):
            msg: ChatMessage = AssistantToolMessage(
                role="assistant",
                content=response["content"],
                tool_calls=tool_calls,
            )
        else:
            msg = Message(role="assistant", content=response["content"] or "")
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str) -> Message:
        """Add a plain text assistant message to the conversation history."""
        msg = Message(role="assistant", content=content)
        self.messages.append(msg)
        return msg

    def add_tool_result(
        self, tool_call_id: str, name: str, content: str
    ) -> ToolResultMessage:
        """Add a tool result message to the conversation history."""
        msg = ToolResultMessage(
            role="tool", tool_call_id=tool_call_id, name=name, content=content
        )
        self.messages.append(msg)
        return msg

    def accumulate_usage(self, usage: UsageToken) -> None:
        """Accumulate usage into total_usage and update last_usage."""
        self.last_usage = usage
        self.total_usage["total"] += usage["total"]
        self.total_usage["input"] += usage["input"]
        self.total_usage["output"] += usage["output"]
        if "cached" in usage:
            self.total_usage["cached"] = (
                self.total_usage.get("cached", 0) + usage["cached"]
            )

    def clear(self) -> None:
        """Clear conversation history, keeping only the system message if set."""
        if self.system:
            self.messages = [Message(role="system", content=self.system)]
        else:
            self.messages = []

    def snapshot(self) -> ConversationSnapshot:
        """Return a serializable snapshot of the full conversation state.

        Includes the system prompt, message history, and accumulated token
        usage. Use with `from_snapshot` to restore.
        """
        return ConversationSnapshot(
            system=self.system,
            messages=list(self.messages),
            total_usage=cast(UsageToken, dict(self.total_usage)),
        )

    @classmethod
    def from_snapshot(cls, data: ConversationSnapshot) -> Self:
        """Rebuild a ConversationState from a `snapshot()` dict.

        Restores the system prompt, message list (kept verbatim — the snapshot
        already contains the system message at index 0 if one was set), and
        accumulated total usage. `last_usage` is not persisted and stays unset.
        """
        state = cls(system=data.get("system"))
        state.messages = list(data.get("messages", []))
        if usage := data.get("total_usage"):
            state.total_usage = cast(UsageToken, dict(usage))
        return state
