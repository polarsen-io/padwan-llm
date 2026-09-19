from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import niquests

Provider = Literal["openai", "gemini", "mistral", "grok", "anthropic", "typesafe"]

__all__ = (
    "LLMError",
    "OutputError",
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


class OutputError(Exception):
    """An `AgentSession` run with `output=` ended without a valid answer."""

    def __init__(self, message: str, *, attempts: int = 0, details: str | None = None):
        self.attempts = attempts
        self.details = details
        super().__init__(message)
