from dataclasses import dataclass, field
from typing import Literal, TypedDict

from .models import UsageToken

__all__ = ("ConversationState", "Message")


class Message(TypedDict):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ConversationState:
    """Pure state container for conversation history and usage tracking.

    Manages message history including an optional system prompt, and tracks
    token usage across multiple turns. This class contains no client or
    provider-specific logic—it's purely state management.
    """

    system: str | None = None
    messages: list[Message] = field(default_factory=list)
    last_usage: UsageToken | None = field(default=None, init=False)
    total_usage: UsageToken = field(
        default_factory=lambda: {"total": 0, "input": 0, "output": 0}, init=False
    )

    def __post_init__(self) -> None:
        if self.system:
            self.messages.insert(0, Message(role="system", content=self.system))

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the conversation history."""
        msg = Message(role="user", content=content)
        self.messages.append(msg)
        return msg

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message to the conversation history."""
        msg = Message(role="assistant", content=content)
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
