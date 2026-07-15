from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import niquests

Provider = Literal["openai", "gemini", "mistral", "grok", "anthropic"]

__all__ = (
    "LLMError",
    "Provider",
    "QuotaExceededError",
    "TooManyRequestsError",
)


class LLMError(Exception):
    def __init__(
        self,
        provider: Provider,
        message: str,
        cause: Exception | None = None,
        body: dict | None = None,
    ):
        self.provider = provider
        self.cause = cause
        self.body = body
        super().__init__(f"[{provider}] {message}")


@dataclasses.dataclass
class TooManyRequestsError(Exception):
    retry_delay: int
    message: str | None = None
    response: niquests.Response | None = None


@dataclasses.dataclass
class QuotaExceededError(Exception):
    body: dict
