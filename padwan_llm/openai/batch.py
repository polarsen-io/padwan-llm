from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from .types import CreateChatCompletionRequest

__all__ = (
    "BatchJob",
    "BatchRequest",
    "BatchResult",
    "TERMINAL_STATUSES",
)

BatchStatus = Literal[
    "validating",
    "failed",
    "in_progress",
    "finalizing",
    "completed",
    "expired",
    "cancelling",
    "cancelled",
]

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)


class RequestCounts(TypedDict):
    total: int
    completed: int
    failed: int


class BatchError(TypedDict):
    code: str
    message: str


@dataclass
class BatchRequest:
    """A single request to include in an OpenAI batch."""

    body: CreateChatCompletionRequest
    custom_id: str | None = None


@dataclass
class BatchJob:
    """Represents an OpenAI batch with its current state."""

    id: str
    status: BatchStatus
    endpoint: str
    input_file_id: str
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int = 0
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    request_counts: RequestCounts | None = None
    metadata: dict[str, str] | None = None
    errors: list[BatchError] | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if batch is in a terminal status."""
        return self.status in TERMINAL_STATUSES

    @property
    def succeeded(self) -> bool:
        """Check if batch completed successfully."""
        return self.status == "completed"

    @classmethod
    def load(cls, data: dict) -> BatchJob:
        """Parse a raw OpenAI batch API response into a BatchJob."""
        errors_data = data.get("errors")
        errors = errors_data.get("data") if errors_data else None
        return cls(
            id=data["id"],
            status=data["status"],
            endpoint=data.get("endpoint", ""),
            input_file_id=data.get("input_file_id", ""),
            output_file_id=data.get("output_file_id"),
            error_file_id=data.get("error_file_id"),
            created_at=data.get("created_at", 0),
            completed_at=data.get("completed_at"),
            failed_at=data.get("failed_at"),
            expired_at=data.get("expired_at"),
            request_counts=data.get("request_counts"),
            metadata=data.get("metadata"),
            errors=errors,
        )


@dataclass
class BatchResult:
    """Parsed result from a single line of the output JSONL file."""

    custom_id: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_line(cls, data: dict) -> BatchResult:
        """Parse a single JSONL line from batch output into a BatchResult.

        Each line has a `custom_id`, a `response` object containing the
        chat completion, and an optional `error` object.
        """
        custom_id = data.get("custom_id", "")
        response = data.get("response", {})
        body = response.get("body", {})

        content = ""
        choices = body.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "") or ""

        usage = body.get("usage", {})

        return cls(
            custom_id=custom_id,
            content=content,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )
