from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Callable, Sequence
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
    "OnThought",
    "Provider",
)

OnThought = Callable[[str], None]


class ChatStream(abc.ABC):
    """Async iterator that yields text chunks and captures usage/tool_calls after iteration."""

    usage: UsageToken | None = None
    tool_calls: list[ToolCall] | None = None

    @abc.abstractmethod
    def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over text chunks from the stream."""
        ...


def to_sse_url(url: str) -> str:
    """Convert http(s):// URL to the niquests SSE scheme (``sse://`` or ``psse://``)."""
    stripped = url.removeprefix("https://")
    if stripped is not url:
        return "sse://" + stripped
    return "psse://" + url.removeprefix("http://")


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
    on_thought: OnThought | None = field(default=None, repr=False)
    """Callback invoked with each chunk of model "thinking" / reasoning text.
    Clients that don't support thinking yet simply never
    invoke it."""

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

    def _sse_url(self, path: str) -> str:
        """Build a full SSE-scheme URL for the given path.

        niquests activates its SSE extension (``r.extension``) only when
        the request URL uses the ``sse://`` (TLS) or ``psse://`` (plain)
        scheme. This helper resolves *path* against ``base_url`` and
        swaps the scheme so streaming requests get proper SSE parsing.
        """
        return to_sse_url(self.base_url.rstrip("/") + "/" + path.lstrip("/"))

    async def __aenter__(self) -> Self:
        if self._session is not None:
            raise RuntimeError(
                f"{type(self).__name__} is already open; "
                "using the same client in nested `async with` blocks is not supported"
            )
        self._session = niquests.AsyncSession(
            timeout=self.timeout, retries=self._retry, base_url=self.base_url
        )
        self._set_auth_headers(self._session)
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
        extra_params: dict[str, Any] | None = None,
    ) -> ChatStream:
        """Stream a chat conversation, yielding text chunks.

        This is a higher-level API than stream() that handles provider-specific
        body building and response extraction. Usage and tool_calls are available
        on the returned ChatStream object after iteration completes.

        The optional ``extra_params`` dict is merged into the request body before
        it is sent, letting callers pass provider-specific fields that are not
        part of the standard interface (e.g. NVIDIA's ``chat_template_kwargs``).
        Providers that don't support extra body fields may silently ignore it.
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
