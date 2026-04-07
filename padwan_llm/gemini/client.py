from __future__ import annotations

import dataclasses
import typing
from dataclasses import field
from functools import partial
import json
import math
import os
from collections.abc import AsyncIterator, Callable, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar, Literal, cast, get_args

import niquests
from urllib3 import HTTPResponse
from urllib3.util.retry import Retry

from .batch import BatchJob, BatchRequest
from .models import (
    BatchJobResponse,
    CompletionBody,
    GenerationConfig,
    ListBatchesResponse,
    StreamBody,
    ThinkingConfig,
)
from .tools import GeminiToolMixin
from .._base import ChatStream, LLMClientBase, Provider
from ..conversation import AssistantToolMessage, ChatMessage, ToolResultMessage
from ..errors import QuotaExceededError, TooManyRequestsError, LLMError
from ..models import (
    ChatResponse,
    FinishReason,
    ToolCall,
    ToolDefinition,
    UsageToken,
)

# Gemini uses SCREAMING_CASE finish reasons; map to our normalized values.
# https://ai.google.dev/api/generate-content#FinishReason
_GEMINI_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "LANGUAGE": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "OTHER": "other",
}

if TYPE_CHECKING:
    from google.genai.types import (
        GenerateContentResponseDict,
        GenerateContentResponseUsageMetadataDict,
        GoogleRpcStatusDict,
    )

GeminiModel = Literal[
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
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


def _raise_error_from_resp(
    data: typing.Any,
    resp: niquests.Response,
    e: niquests.exceptions.HTTPError,
) -> typing.Never:
    """Raise the appropriate error given parsed response data."""
    error = cast("GoogleRpcStatusDict", data.get("error", {}))
    if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
        retry_delay: int | None = None
        for detail in error.get("details") or []:
            if detail.get("@type") == "type.googleapis.com/google.rpc.QuotaFailure":
                raise QuotaExceededError(body=data)
            if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                retry_delay = _parse_retry_delay(detail["retryDelay"])
        if retry_delay is None:
            raise e
        raise TooManyRequestsError(retry_delay=retry_delay, response=resp)
    raise LLMError(
        "gemini", f"{resp.status_code} {error.get('message', '')}", body=data
    ) from e


def _check_resp_status(resp: niquests.Response) -> niquests.Response:
    """Check HTTP status and raise appropriate errors without consuming the body."""
    try:
        return resp.raise_for_status()
    except niquests.exceptions.HTTPError as e:
        try:
            data = resp.json()
        except Exception:
            raise e
        _raise_error_from_resp(data, resp, e)


async def _async_check_resp_status(
    resp: niquests.AsyncResponse,
) -> niquests.AsyncResponse:
    """Check HTTP status for a streaming response, raising appropriate errors."""
    try:
        return resp.raise_for_status()  # type: ignore[return-value]
    except niquests.exceptions.HTTPError as e:
        try:
            data = await resp.json()
        except Exception:
            raise e
        _raise_error_from_resp(data, resp, e)


def _check_resp(resp: niquests.Response) -> typing.Any:
    """Check Gemini HTTP response and return the parsed JSON body."""
    return (_check_resp_status(resp)).json()


def is_gemini_model(model_name: str | None) -> bool:
    """Check if the model name is a Gemini model."""
    if model_name is None:
        return False
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
        except (json.JSONDecodeError, ValueError, KeyError):  # fmt: skip
            pass
        return None


@dataclasses.dataclass
class GeminiClient(LLMClientBase[GeminiRetry], GeminiToolMixin):
    """Gemini API client with structured output support."""

    provider: ClassVar[Provider] = "gemini"
    model: str | None = "gemini-2.5-flash"
    base_url: str = GEMINI_ENDPOINT
    on_thought: Callable[[str], None] | None = None
    thinking_config: ThinkingConfig | None = None
    _retry: GeminiRetry = field(
        default_factory=partial(
            GeminiRetry,
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
        )
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
        if not _model:
            raise LLMError(self.provider, "No model specified")
        resp = await self.session.post(
            f"/models/{_model}:generateContent",
            json=body,
        )
        data: GenerateContentResponseDict = _check_resp(resp)

        usage: GenerateContentResponseUsageMetadataDict = data.get("usageMetadata", {})
        token: UsageToken = {
            "total": usage.get("totalTokenCount", 0),
            "input": usage.get("promptTokenCount", 0),
            "output": usage.get("candidatesTokenCount", 0),
        }
        if (cached := usage.get("cachedContentTokenCount")) is not None:
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
        if not _model:
            raise LLMError(self.provider, "No model specified")
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
        data: BatchJobResponse = _check_resp(resp)
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
        data: BatchJobResponse = _check_resp(resp)
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
        data: ListBatchesResponse = _check_resp(resp)

        jobs = [self._parse_batch_job(op) for op in data.get("operations", [])]
        return jobs, data.get("nextPageToken")

    async def cancel_batch(self, job_name: str) -> BatchJobResponse:
        """Cancel a pending or running batch job.

        Args:
            job_name: Batch job name (e.g., "batches/123456" or just "123456")
        """
        if not job_name.startswith("batches/"):
            job_name = f"batches/{job_name}"

        resp = await self.session.post(f"/{job_name}:cancel")
        data: BatchJobResponse = _check_resp(resp)
        return data

    def _parse_batch_job(self, data: BatchJobResponse) -> BatchJob:
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
        if not self.model:
            raise LLMError(self.provider, "No model specified for streaming")
        _temperature = body.get("temperature", self.temperature)

        gen_config: GenerationConfig = {"temperature": _temperature}
        if self.thinking_config:
            gen_config["thinkingConfig"] = self.thinking_config
        payload: dict = {
            "contents": body["contents"],
            "generationConfig": gen_config,
        }
        if system := body.get("systemInstruction"):
            payload["systemInstruction"] = system
        if gemini_tools := body.get("tools"):
            payload["tools"] = gemini_tools

        resp = await self.session.post(
            f"/models/{self.model}:streamGenerateContent",
            params={"alt": "sse"},
            json=payload,
            stream=True,
        )
        await _async_check_resp_status(resp)
        resp.encoding = "utf-8"

        async for line in resp.iter_lines(decode_unicode=True):  # pyright: ignore[reportGeneralTypeIssues]
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError as e:
                    raise LLMError(self.provider, f"Stream parse error: {e}") from e

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> tuple[ChatResponse, UsageToken]:
        """Send a chat conversation and return the structured response."""
        stream_body = GeminiChatStream.build_body(messages, self.temperature, tools)
        body: CompletionBody = {
            "contents": stream_body["contents"],
            "generationConfig": {"temperature": self.temperature},
        }
        if system := stream_body.get("systemInstruction"):
            body["systemInstruction"] = system
        if gemini_tools := stream_body.get("tools"):
            body["tools"] = gemini_tools
        data, token = await self.complete(body, self.model)
        candidates = data.get("candidates")
        if not candidates or not (content := candidates[0].get("content")):
            raise LLMError(self.provider, "No content in response")
        parts = cast(list[dict[str, typing.Any]], content.get("parts") or [])

        raw_reason = candidates[0].get("finishReason", "STOP")
        tool_calls = self._extract_gemini_tool_calls(parts)

        # Extract text from parts (may coexist with function calls)
        text: str | None = None
        for part in parts:
            if (t := part.get("text")) is not None:
                text = t
                break

        if tool_calls:
            finish_reason: FinishReason = "tool_calls"
        else:
            finish_reason = _GEMINI_FINISH_REASON_MAP.get(raw_reason, "other")

        response: ChatResponse = {
            "content": text,
            "finish_reason": finish_reason,
        }
        if tool_calls:
            response["tool_calls"] = tool_calls
        if text is None and not tool_calls:
            raise LLMError(self.provider, "No text content in response")
        return response, token

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> GeminiChatStream:
        """Stream a chat conversation, yielding text chunks.

        Handles Gemini-specific body building and response extraction.
        Usage and tool_calls are available on the returned stream object after iteration.
        """
        return GeminiChatStream(self, messages, tools, on_thought=self.on_thought)


class GeminiChatStream(ChatStream, GeminiToolMixin):
    """ChatStream implementation for Gemini API."""

    def __init__(
        self,
        client: GeminiClient,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
        on_thought: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self._messages = messages
        self._tools = tools
        self._on_thought = on_thought
        self.usage: UsageToken | None = None
        self.tool_calls: list[ToolCall] | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        body = self.build_body(self._messages, self._client.temperature, self._tools)
        pending_tool_calls: list[ToolCall] = []
        async for chunk in self._client.stream(body):
            if text := self._extract_text(chunk):
                yield text
            new_calls = self._extract_tool_calls_from_chunk(
                chunk, id_offset=len(pending_tool_calls)
            )
            pending_tool_calls.extend(new_calls)
            if usage := self._extract_usage(chunk):
                self.usage = usage
        if pending_tool_calls:
            self.tool_calls = pending_tool_calls

    @staticmethod
    def build_body(
        messages: Sequence[ChatMessage],
        temperature: float,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> StreamBody:
        """Build Gemini request body from messages.

        Gemini uses 'model' instead of 'assistant' and wraps content in parts.
        System messages are translated to Gemini's systemInstruction field.
        Handles AssistantToolMessage and ToolResultMessage for tool flows.
        """
        contents = []
        system_parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_parts.append(msg["content"])  # type: ignore[typeddict-item]
                continue
            if role == "tool":
                contents.append(
                    GeminiToolMixin._convert_tool_result(cast(ToolResultMessage, msg))
                )
                continue
            if role == "assistant" and "tool_calls" in msg:
                contents.append(
                    GeminiToolMixin._convert_assistant_tool_message(
                        cast(AssistantToolMessage, msg)
                    )
                )
                continue
            # Regular text message (user or assistant)
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": msg["content"]}]})  # type: ignore[typeddict-item]
        body: StreamBody = {"contents": contents, "temperature": temperature}
        if system_parts:
            body["systemInstruction"] = {
                "parts": [{"text": text} for text in system_parts]
            }
        if tools:
            body["tools"] = GeminiToolMixin._tools_to_gemini(tools)
        return body

    @staticmethod
    def _extract_tool_calls_from_chunk(
        chunk: dict, id_offset: int = 0
    ) -> list[ToolCall]:
        """Extract function call tool calls from a Gemini stream chunk."""
        candidates = chunk.get("candidates")
        if not candidates:
            return []
        content = candidates[0].get("content")
        if not content:
            return []
        parts = content.get("parts", [])
        return GeminiToolMixin._extract_gemini_tool_calls(parts, id_offset)

    def _extract_text(self, chunk: dict) -> str | None:
        """Extract text content from a Gemini stream response chunk.

        Thought parts are forwarded to `on_thought` if set and skipped
        as regular text; callers collect them via the callback.
        """
        if candidates := chunk.get("candidates"):
            if content := candidates[0].get("content"):
                for part in content.get("parts", []):
                    if part.get("thought"):
                        if self._on_thought and (thought_text := part.get("text")):
                            self._on_thought(thought_text)
                        continue
                    if (text := part.get("text")) is not None:
                        return text
        return None

    def _extract_usage(self, chunk: dict) -> UsageToken | None:
        """Extract usage info from a Gemini stream response chunk."""
        if usage := chunk.get("usageMetadata"):
            token: UsageToken = {
                "total": usage.get("totalTokenCount", 0),
                "input": usage.get("promptTokenCount", 0),
                "output": usage.get("candidatesTokenCount", 0),
            }
            if (cached := usage.get("cachedContentTokenCount")) is not None:
                token["cached"] = cached
            return token
        return None
