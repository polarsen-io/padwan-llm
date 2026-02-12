from __future__ import annotations

import dataclasses
import json
import math
import os
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar, Literal, cast, get_args

import niquests
from urllib3 import HTTPResponse
from urllib3.util.retry import Retry

from .batch import BatchJob, BatchRequest
from .models import CompletionBody, StreamBody
from .._base import ChatStream, LLMClientBase, Provider
from ..conversation import Message
from ..errors import QuotaExceededError, TooManyRequestsError, LLMError
from ..models import UsageToken

if TYPE_CHECKING:
    from google.genai.types import (
        GenerateContentResponseDict,
        GenerateContentResponseUsageMetadataDict,
    )

GeminiModel = Literal[
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

__all__ = (
    "GeminiClient",
    "GEMINI_MODELS",
    "GEMINI_ENDPOINT",
    "GeminiModel",
    "is_gemini_model",
)


def _parse_retry_delay(delay_str: str) -> int:
    """Parse retry delay string like '48s' or '48.24916365s' to int seconds (rounded up)."""
    return math.ceil(float(delay_str.rstrip("s")))


def _check_resp(resp: niquests.Response) -> dict:
    """Check Gemini HTTP response, extracting retry delay from body."""
    try:
        resp.raise_for_status()
    except niquests.exceptions.HTTPError as e:
        data = resp.json()
        if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            retry_delay: int | None = None
            for detail in data.get("error", {}).get("details", []):
                if detail.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
                    raise QuotaExceededError(body=data)
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    retry_delay = _parse_retry_delay(detail["retryDelay"])
            if retry_delay is None:
                raise e
            raise TooManyRequestsError(retry_delay=retry_delay, response=resp)
        error_msg = data.get("error", {}).get("message", "")
        raise LLMError("gemini", f"{resp.status_code} {error_msg}") from e
    return resp.json()


def is_gemini_model(model_name: str) -> bool:
    """Check if the model name is a Gemini model."""
    return model_name.startswith("gemini")


GEMINI_MODELS: set[str] = set(get_args(GeminiModel))

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/"

# State mapping: API returns BATCH_STATE_*, we normalize to JOB_STATE_*
_BATCH_STATE_MAP: dict[str, str] = {
    "BATCH_STATE_UNSPECIFIED": "JOB_STATE_UNSPECIFIED",
    "BATCH_STATE_PENDING": "JOB_STATE_PENDING",
    "BATCH_STATE_QUEUED": "JOB_STATE_QUEUED",
    "BATCH_STATE_RUNNING": "JOB_STATE_RUNNING",
    "BATCH_STATE_SUCCEEDED": "JOB_STATE_SUCCEEDED",
    "BATCH_STATE_FAILED": "JOB_STATE_FAILED",
    "BATCH_STATE_CANCELLING": "JOB_STATE_CANCELLING",
    "BATCH_STATE_CANCELLED": "JOB_STATE_CANCELLED",
    "BATCH_STATE_EXPIRED": "JOB_STATE_EXPIRED",
}


class GeminiRetry(Retry):
    """Custom retry that extracts retry delay from Gemini response body."""

    def get_retry_after(self, response: HTTPResponse) -> float | None:
        """Extract retry delay from response body (google.rpc.RetryInfo)."""
        try:
            data = json.loads(response.data.decode("utf-8"))
            for detail in data.get("error", {}).get("details", []):
                if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    delay_str = detail.get("retryDelay", "")
                    return math.ceil(float(delay_str.rstrip("s")))
        except json.JSONDecodeError, ValueError, KeyError:
            pass
        return None


@dataclasses.dataclass
class GeminiClient(LLMClientBase[GeminiRetry]):
    """Gemini API client with structured output support."""

    provider: ClassVar[Provider] = "gemini"
    model: str | None = "gemini-2.0-flash"
    base_url: str = GEMINI_ENDPOINT
    _retry: GeminiRetry = GeminiRetry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    def _get_default_api_key(self) -> str:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError(self.provider, "GEMINI_API_KEY not set")
        return api_key

    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        session.headers["x-goog-api-key"] = self._api_key

    async def complete(
        self, body: CompletionBody, model: str | None = None
    ) -> tuple[GenerateContentResponseDict, UsageToken]:
        """Fetch structured completion from Gemini."""
        _model = model or self.model
        resp = await self.session.post(
            f"/models/{_model}:generateContent",
            json=body,
        )
        data = cast("GenerateContentResponseDict", _check_resp(resp))

        usage: GenerateContentResponseUsageMetadataDict = data.get("usageMetadata", {})
        token: UsageToken = {
            "total": usage.get("totalTokenCount", 0),
            "input": usage.get("promptTokenCount", 0),
            "output": usage.get("candidatesTokenCount", 0),
        }
        if cached := usage.get("cachedContentTokenCount"):
            token["cached"] = cached

        return data, token

    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str | None = None,
        display_name: str | None = None,
    ) -> BatchJob:
        """Create a batch job for multiple requests.

        Args:
            requests: List of BatchRequest objects to process
            model: Model to use (defaults to self.model)
            display_name: Optional display name for the batch job

        Returns:
            BatchJob with the job name for polling via get_batch().
        """
        _model = model or self.model
        batch_requests = []

        for idx, req in enumerate(requests):
            request_payload: dict = {"contents": req.contents}
            if req.generation_config:
                request_payload["generationConfig"] = req.generation_config
            if req.system_instruction:
                request_payload["systemInstruction"] = req.system_instruction

            batch_requests.append(
                {
                    "request": request_payload,
                    "metadata": {"key": req.key or f"request-{idx}"},
                }
            )

        payload = {
            "batch": {
                "displayName": display_name or "batch-request",
                "inputConfig": {
                    "requests": {"requests": batch_requests},
                },
            },
        }

        resp = await self.session.post(
            f"/models/{_model}:batchGenerateContent",
            json=payload,
        )
        data = _check_resp(resp)
        return self._parse_batch_job(data)

    async def get_batch(self, job_name: str) -> BatchJob:
        """Get the current status of a batch job.

        Args:
            job_name: Batch job name (e.g., "batches/123456" or just "123456")

        Returns:
            BatchJob with current state and results if completed.
        """
        if not job_name.startswith("batches/"):
            job_name = f"batches/{job_name}"

        resp = await self.session.get(f"/{job_name}")
        data = _check_resp(resp)
        return self._parse_batch_job(data)

    async def list_batches(
        self,
        page_size: int = 10,
        page_token: str | None = None,
    ) -> tuple[list[BatchJob], str | None]:
        """List batch jobs with pagination.

        Args:
            page_size: Maximum number of jobs to return (max 100)
            page_token: Token from previous response for pagination

        Returns:
            Tuple of (jobs list, next_page_token or None).
        """
        params: dict[str, str] = {"pageSize": str(min(page_size, 100))}
        if page_token:
            params["pageToken"] = page_token

        resp = await self.session.get("/batches", params=params)
        data = _check_resp(resp)

        jobs = [self._parse_batch_job(op) for op in data.get("operations", [])]
        return jobs, data.get("nextPageToken")

    async def cancel_batch(self, job_name: str) -> None:
        """Cancel a pending or running batch job.

        Args:
            job_name: Batch job name (e.g., "batches/123456" or just "123456")
        """
        if not job_name.startswith("batches/"):
            job_name = f"batches/{job_name}"

        resp = await self.session.post(f"/{job_name}:cancel")
        _check_resp(resp)

    def _parse_batch_job(self, data: dict) -> BatchJob:
        """Parse raw API response into BatchJob dataclass."""
        from .models import BatchDestination, BatchState

        metadata = data.get("metadata", {})
        output = metadata.get("output", {})

        # Build destination from output
        dest: BatchDestination | None = None
        if output:
            dest_dict: BatchDestination = {}
            if "responsesFile" in output:
                dest_dict["file_name"] = output["responsesFile"]
            if inlined := output.get("inlinedResponses", {}).get("inlinedResponses"):
                dest_dict["inlined_responses"] = inlined
            if dest_dict:
                dest = dest_dict

        # Transform state from BATCH_STATE_* to JOB_STATE_*
        raw_state = metadata.get("state", "BATCH_STATE_UNSPECIFIED")
        state = cast(BatchState, _BATCH_STATE_MAP.get(raw_state, raw_state))

        return BatchJob(
            name=data.get("name", ""),
            state=state,
            display_name=metadata.get("displayName"),
            model=metadata.get("model"),
            create_time=metadata.get("createTime"),
            start_time=metadata.get("startTime"),
            end_time=metadata.get("endTime"),
            update_time=metadata.get("updateTime"),
            error=data.get("error"),
            dest=dest,
            stats=metadata.get("batchStats"),
        )

    async def stream(self, body: StreamBody) -> AsyncIterator[dict]:
        """Stream chat completions from Gemini, yielding response chunks as they arrive via SSE."""
        _temperature = body.get("temperature", self.temperature)

        resp = await self.session.post(
            f"/models/{self.model}:streamGenerateContent",
            params={"alt": "sse"},
            json={
                "contents": body["contents"],
                "generationConfig": {"temperature": _temperature},
            },
            stream=True,
        )
        resp.raise_for_status()
        resp.encoding = "utf-8"

        async for line in resp.iter_lines(decode_unicode=True):  # pyright: ignore[reportGeneralTypeIssues]
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                yield json.loads(data_str)

    async def complete_chat(self, messages: list[Message]) -> tuple[str, UsageToken]:
        """Send a chat conversation and return the complete response."""
        stream_body = GeminiChatStream.build_body(messages, self.temperature)
        body: CompletionBody = {
            "contents": stream_body["contents"],
            "generationConfig": {"temperature": self.temperature},
        }
        data, token = await self.complete(body, self.model)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text, token

    def stream_chat(self, messages: list[Message]) -> GeminiChatStream:
        """Stream a chat conversation, yielding text chunks.

        Handles Gemini-specific body building and response extraction.
        Usage is available on the returned stream object after iteration.
        """
        return GeminiChatStream(self, messages)


class GeminiChatStream(ChatStream):
    """ChatStream implementation for Gemini API."""

    def __init__(self, client: GeminiClient, messages: list[Message]) -> None:
        self._client = client
        self._messages = messages
        self.usage: UsageToken | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        body = self.build_body(self._messages, self._client.temperature)
        async for chunk in self._client.stream(body):
            if text := self._extract_text(chunk):
                yield text
            if usage := self._extract_usage(chunk):
                self.usage = usage

    @staticmethod
    def build_body(messages: list[Message], temperature: float) -> StreamBody:
        """Build Gemini request body from messages.

        Gemini uses 'model' instead of 'assistant' and wraps content in parts.
        System messages are skipped here (Gemini handles them separately).
        """
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        return {"contents": contents, "temperature": temperature}

    def _extract_text(self, chunk: dict) -> str | None:
        """Extract text content from a Gemini stream response chunk."""
        if candidates := chunk.get("candidates"):
            if content := candidates[0].get("content"):
                if parts := content.get("parts"):
                    return parts[0].get("text")
        return None

    def _extract_usage(self, chunk: dict) -> UsageToken | None:
        """Extract usage info from a Gemini stream response chunk."""
        if usage := chunk.get("usageMetadata"):
            token: UsageToken = {
                "total": usage.get("totalTokenCount", 0),
                "input": usage.get("promptTokenCount", 0),
                "output": usage.get("candidatesTokenCount", 0),
            }
            if cached := usage.get("cachedContentTokenCount"):
                token["cached"] = cached
            return token
        return None
