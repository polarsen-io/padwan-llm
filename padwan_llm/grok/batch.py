from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..openai.types import CreateChatCompletionRequest

__all__ = (
    "GrokBatchJob",
    "GrokBatchRequest",
    "GrokBatchResult",
)


@dataclass
class GrokBatchRequest:
    """A single request to include in a Grok batch.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#add-requests-to-a-batch
    """

    body: CreateChatCompletionRequest
    custom_id: str | None = None


@dataclass
class GrokBatchJob:
    """Represents a Grok batch with its current state.

    The xAI batch API tracks progress via counters rather than a single status
    string. A batch is terminal when no requests are pending (and at least one
    request has been submitted).

    https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-details
    """

    batch_id: str
    name: str = ""
    num_requests: int = 0
    num_pending: int = 0
    num_success: int = 0
    num_error: int = 0
    num_cancelled: int = 0
    create_time: str | None = None
    expire_time: str | None = None

    @property
    def is_terminal(self) -> bool:
        """Check if all requests have been processed."""
        return self.num_pending == 0 and self.num_requests > 0

    @property
    def succeeded(self) -> bool:
        """Check if all requests completed successfully."""
        return self.is_terminal and self.num_error == 0 and self.num_cancelled == 0

    @classmethod
    def load(cls, data: dict) -> GrokBatchJob:
        """Parse a raw xAI batch API response into a GrokBatchJob."""
        state = data.get("state", {})
        return cls(
            batch_id=data["batch_id"],
            name=data.get("name", ""),
            num_requests=state.get("num_requests", 0),
            num_pending=state.get("num_pending", 0),
            num_success=state.get("num_success", 0),
            num_error=state.get("num_error", 0),
            num_cancelled=state.get("num_cancelled", 0),
            create_time=data.get("create_time"),
            expire_time=data.get("expire_time"),
        )


@dataclass
class GrokBatchResult:
    """Parsed result from a Grok batch response.

    Each result wraps a full OpenAI-compatible chat completion response
    nested under ``batch_result.response.chat_get_completion``. Failed
    results have ``error_message`` set and empty ``content``.

    https://docs.x.ai/developers/rest-api-reference/inference/batches#get-batch-results
    """

    custom_id: str
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str | None = None
    error_message: str | None = None

    @classmethod
    def from_response(cls, data: dict) -> GrokBatchResult:
        """Parse a single entry from the batch results list.

        The response is nested as
        ``data.batch_result.response.chat_get_completion`` and contains a
        standard OpenAI chat completion with ``choices`` and ``usage``.
        """
        batch_result = data.get("batch_result", {})
        error = batch_result.get("error")

        response = batch_result.get("response", {})
        completion = response.get("chat_get_completion", {})

        content = ""
        finish_reason = None
        choices = completion.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "") or ""
            finish_reason = choices[0].get("finish_reason")

        usage = completion.get("usage", {})

        return cls(
            custom_id=data["batch_request_id"],
            content=content,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            finish_reason=finish_reason,
            error_message=(error.get("message") if isinstance(error, dict) else error)
            if error
            else None,
        )
