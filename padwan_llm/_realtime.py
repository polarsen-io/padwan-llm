import abc
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .errors import Provider

__all__ = ("RealtimeClientBase",)


@dataclass
class RealtimeClientBase(abc.ABC):
    """Abstract base for realtime speech-to-speech clients over a WebSocket.

    Use provider-specific clients (OpenAIRealtimeClient, ...) or the
    :func:`padwan_llm.RealtimeClient` factory instead.
    """

    provider: ClassVar[Provider]

    model: str
    api_key: str | None = field(default=None, repr=False)
    base_url: str = ""
    timeout: float = 30.0

    _api_key: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._api_key = self.api_key or self._get_default_api_key()

    @abc.abstractmethod
    def _get_default_api_key(self) -> str:
        """Get API key from environment."""
        ...

    @abc.abstractmethod
    def connect(self) -> AbstractAsyncContextManager[Any]:
        """Open a configured realtime session and yield a live connection."""
        ...
