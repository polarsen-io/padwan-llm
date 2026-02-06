from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from .._base import LLMClientBase
from ..conversation import ConversationState, Message
from ..models import UsageToken

__all__ = ("Conversation", "Message")


@dataclass
class Conversation:
    """Manages a multi-turn conversation with an LLM client.

    The Conversation class wraps ConversationState and provides streaming
    interface for chat interactions. Provider-specific details are handled
    by the client's stream_chat method.

    Example:
        async with LLMClient(model="gpt-4o") as client:
            conv = Conversation(client, system="You are a helpful assistant.")
            async for chunk in conv.stream("Hello!"):
                print(chunk, end="", flush=True)
    """

    client: LLMClientBase
    system: str | None = None
    _state: ConversationState = field(init=False)

    def __post_init__(self) -> None:
        self._state = ConversationState(system=self.system)

    @property
    def messages(self) -> list[Message]:
        """Access the conversation message history."""
        return self._state.messages

    @messages.setter
    def messages(self, value: list[Message]) -> None:
        """Set the conversation message history."""
        self._state.messages = value

    @property
    def last_usage(self) -> UsageToken | None:
        """Token usage from the last request."""
        return self._state.last_usage

    @property
    def total_usage(self) -> UsageToken:
        """Cumulative token usage across all requests."""
        return self._state.total_usage

    def add_user_message(self, content: str) -> Message:
        """Add a user message to the conversation history."""
        return self._state.add_user_message(content)

    def add_assistant_message(self, content: str) -> Message:
        """Add an assistant message to the conversation history."""
        return self._state.add_assistant_message(content)

    def clear(self) -> None:
        """Clear conversation history, keeping only the system message if set."""
        self._state.clear()

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Send a user message and stream the assistant's response.

        Adds the user message to history, streams the response via the client's
        stream_chat method, yields text as it arrives, and saves the complete
        response to history when done. Also tracks token usage.
        """
        self.add_user_message(user_input)

        response_chunks: list[str] = []
        chat_stream = self.client.stream_chat(list(self.messages))

        async for text in chat_stream:
            response_chunks.append(text)
            yield text

        if chat_stream.usage:
            self._state.accumulate_usage(chat_stream.usage)

        full_response = "".join(response_chunks)
        self.add_assistant_message(full_response)

    async def send(self, user_input: str) -> str:
        """Send a user message and return the complete response (non-streaming)."""
        chunks = []
        async for chunk in self.stream(user_input):
            chunks.append(chunk)
        return "".join(chunks)
