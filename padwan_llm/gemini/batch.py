"""Batch processing dataclasses for Gemini API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import (
        BatchDestination,
        BatchState,
        BatchStats,
        Content,
        GenerationConfig,
        InlinedResponse,
        SystemInstruction,
    )

__all__ = (
    "BatchJob",
    "BatchRequest",
    "BatchResult",
    "TERMINAL_STATES",
)

TERMINAL_STATES: frozenset[BatchState] = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)


@dataclass
class BatchJob:
    """Represents a Gemini batch job with its current state."""

    name: str
    state: BatchState
    display_name: str | None = None
    model: str | None = None
    create_time: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    update_time: str | None = None
    error: dict | None = None
    dest: BatchDestination | None = None
    stats: BatchStats | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        """Check if job completed successfully."""
        return self.state == "JOB_STATE_SUCCEEDED"

    @property
    def inlined_responses(self) -> list[InlinedResponse] | None:
        """Get inlined responses if available."""
        if self.dest is None:
            return None
        return self.dest.get("inlined_responses")


@dataclass
class BatchRequest:
    """A single request to include in a batch."""

    contents: list[Content]
    generation_config: GenerationConfig | None = None
    system_instruction: SystemInstruction | None = None
    key: str | None = None


@dataclass
class BatchResult:
    """Result of a single request from a completed batch."""

    key: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_inlined_response(cls, resp: InlinedResponse) -> BatchResult:
        """Parse a single inlined response from batch results."""
        key = resp.get("key", "")
        response = resp.get("response", {})

        # Extract text content from response
        content = ""
        candidates = response.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                content = parts[0].get("text", "")

        # Extract usage metadata
        usage = response.get("usageMetadata", {})

        return cls(
            key=key,
            content=content,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            total_tokens=usage.get("totalTokenCount", 0),
        )
