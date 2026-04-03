from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Self

import niquests
from urllib3.util.retry import Retry

from .conversation import ChatMessage
from .errors import LLMError, Provider
from .models import ChatResponse, ToolCall, ToolDefinition, UsageToken

if TYPE_CHECKING:
    from types import TracebackType

__all__ = (
    "ChatStream",
    "LLMClientBase",
    "Provider",
)


class ChatStream(abc.ABC):
    """Async iterator that yields text chunks and captures usage/tool_calls after iteration."""

    usage: UsageToken | None = None
    tool_calls: list[ToolCall] | None = None

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over text chunks from the stream."""
        ...


@dataclass
class LLMClientBase[RetryT: Retry](abc.ABC):
    """Abstract base class for LLM provider clients.

    Use provider-specific clients (OpenAIClient, GeminiClient, etc.) instead.
    """

    provider: ClassVar[Provider]
    _retry: RetryT

    model: str | None = None
    """Default model to use (e.g., 'gpt-4o', 'gemini-2.5-flash')."""
    temperature: float = 0.2
    """Sampling temperature."""
    timeout: float = 60
    """Request timeout in seconds."""
    api_key: str | None = field(default=None, repr=False)
    """API key. If None, reads from provider's environment variable."""

    _api_key: str = field(init=False, repr=False)
    base_url: str = field(init=False, default="", repr=False)
    _session: niquests.AsyncSession | None = field(init=False, default=None, repr=False)

    @property
    def max_retries(self) -> int | None:
        """Maximum retry attempts, derived from _retry config. None means unlimited, 0 means disabled."""
        total = self._retry.total
        if total is False:
            return 0
        if total is None:
            return None
        return int(total)

    def __post_init__(self) -> None:
        self._api_key = self.api_key or self._get_default_api_key()

    @abc.abstractmethod
    def _get_default_api_key(self) -> str:
        """Get API key from environment."""
        ...

    @abc.abstractmethod
    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        """Set provider-specific authentication headers."""
        ...

    @property
    def session(self) -> niquests.AsyncSession:
        """Get active session, raising if not initialized."""
        if self._session is None:
            raise LLMError(
                self.provider, "Client not initialized. Use async context manager."
            )
        return self._session

    def _build_session(self) -> niquests.AsyncSession:
        """Create and configure an async HTTP session with retry logic."""
        session = niquests.AsyncSession(
            timeout=self.timeout, retries=self._retry, base_url=self.base_url
        )
        self._set_auth_headers(session)
        return session

    async def __aenter__(self) -> Self:
        self._session = self._build_session()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self._session:
            try:
                await self._session.close()
            finally:
                self._session = None

    @abc.abstractmethod
    async def complete(self, body: Any) -> tuple[Any, UsageToken]:
        """Fetch a structured completion from the provider.

        Each provider defines its own body and response types.
        """
        ...

    @abc.abstractmethod
    def stream(self, body: Any) -> AsyncIterator[Any]:
        """Stream chat completions, yielding response chunks as they arrive.

        Each provider defines its own body and response types.
        """
        ...

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> ChatStream:
        """Stream a chat conversation, yielding text chunks.

        This is a higher-level API than stream() that handles provider-specific
        body building and response extraction. Usage and tool_calls are available
        on the returned ChatStream object after iteration completes.
        """
        ...

    @abc.abstractmethod
    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> tuple[ChatResponse, UsageToken]:
        """Send a chat conversation and return the structured response.

        Returns a ChatResponse with content, optional tool_calls, and finish_reason,
        along with token usage.
        """
        ...
