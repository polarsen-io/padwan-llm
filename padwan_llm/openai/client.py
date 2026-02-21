import dataclasses
import typing
from dataclasses import field
from functools import partial
import json
import os
from collections.abc import AsyncIterator, Sequence
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar, Literal, cast, get_args

import niquests
from urllib3.util.retry import Retry

from .batch import BatchJob, BatchRequest, BatchResult
from .tools import OpenAIToolMixin
from .types import Batch, ListBatchesResponse, OpenAIFile
from .._base import ChatStream, LLMClientBase, LLMError, Provider
from ..conversation import ChatMessage
from ..errors import QuotaExceededError, TooManyRequestsError
from ..models import (
    ChatResponse,
    FinishReason,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    UsageToken,
)

if TYPE_CHECKING:
    from .types import (
        CompletionUsage,
        CreateChatCompletionRequest,
        CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse,
    )

OpenAIModel = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
]

__all__ = (
    "_OpenAIBase",
    "OpenAIClient",
    "OPENAI_MODELS",
    "OPENAI_ENDPOINT",
    "OpenAIModel",
    "is_openai_model",
)


def _check_resp_status(resp: niquests.Response) -> niquests.Response:
    """Check HTTP status and raise appropriate errors without consuming the body."""
    try:
        return resp.raise_for_status()
    except niquests.exceptions.HTTPError as e:
        try:
            data = resp.json()
        except Exception:
            raise e
        if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            retry_after = resp.headers.get("retry-after")
            retry_delay = int(retry_after) if retry_after else 60
            raise TooManyRequestsError(retry_delay=retry_delay, response=resp)
        if resp.status_code == HTTPStatus.PAYMENT_REQUIRED:
            raise QuotaExceededError(body=data)
        error = data.get("error", "") if isinstance(data, dict) else data
        msg = error.get("message", "") if isinstance(error, dict) else str(error)
        raise LLMError("openai", f"{resp.status_code} {msg}", body=data) from e


def _check_resp(resp: niquests.Response) -> typing.Any:
    """Check OpenAI-compatible HTTP response and return the parsed JSON body."""
    return (_check_resp_status(resp)).json()


_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def is_openai_model(model_name: str | None) -> bool:
    """Check if the model name looks like an OpenAI model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith(_OPENAI_PREFIXES)


OPENAI_MODELS: set[str] = set(get_args(OpenAIModel))

OPENAI_ENDPOINT = "https://api.openai.com/v1/"


@dataclasses.dataclass
class _OpenAIBase(LLMClientBase[Retry], OpenAIToolMixin):
    """OpenAI-compatible base client with chat and auth logic.

    Shared by OpenAIClient, GrokClient, and MistralClient. Does not include
    batch methods — those live on the concrete provider clients.
    """

    provider: ClassVar[Provider] = "openai"
    model: str | None = "gpt-4o"
    base_url: str = OPENAI_ENDPOINT
    _retry: Retry = field(
        default_factory=partial(
            Retry,
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"],
        )
    )

    def _get_default_api_key(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError(self.provider, "OPENAI_API_KEY not set")
        return api_key

    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        session.headers["Authorization"] = f"Bearer {self._api_key}"

    async def complete(
        self,
        body: CreateChatCompletionRequest,
    ) -> tuple[CreateChatCompletionResponse, UsageToken]:
        """Fetch structured completion from an OpenAI-compatible endpoint."""
        resp = await self.session.post(
            "/chat/completions",
            json=body,
        )
        data: CreateChatCompletionResponse = _check_resp(resp)

        usage: CompletionUsage | None = data.get("usage")
        if usage is None:
            raise LLMError(self.provider, "No usage in response")
        token: UsageToken = {
            "total": usage["total_tokens"],
            "input": usage["prompt_tokens"],
            "output": usage["completion_tokens"],
        }
        if details := usage.get("prompt_tokens_details"):
            if (cached := details.get("cached_tokens")) is not None:
                token["cached"] = cached

        return data, token

    async def stream(
        self,
        body: CreateChatCompletionRequest,
    ) -> AsyncIterator[CreateChatCompletionStreamResponse]:
        """Stream chat completions, yielding response chunks as they arrive via SSE."""
        resp = await self.session.post(
            "/chat/completions",
            json={**body, "stream": True},
            stream=True,
        )
        _check_resp_status(resp)

        async for raw_line in resp.iter_lines(decode_unicode=True):  # pyright: ignore[reportGeneralTypeIssues]
            if not raw_line:
                continue
            line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    yield cast(
                        "CreateChatCompletionStreamResponse", json.loads(data_str)
                    )
                except json.JSONDecodeError as e:
                    raise LLMError(self.provider, f"Stream parse error: {e}") from e

    async def complete_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> tuple[ChatResponse, UsageToken]:
        """Send a chat conversation and return the structured response."""
        if not self.model:
            raise LLMError(self.provider, "No model specified")
        body: CreateChatCompletionRequest = {
            "model": self.model,
            "messages": cast(list, messages),
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = cast(list, self._tools_to_openai(tools))
        data, token = await self.complete(body)
        choice = data["choices"][0]
        message = choice["message"]
        response: ChatResponse = {
            "content": message.get("content"),
            "finish_reason": cast("FinishReason", choice["finish_reason"]),
        }
        if raw_tool_calls := message.get("tool_calls"):
            response["tool_calls"] = self._extract_tool_calls(
                cast(list, raw_tool_calls)
            )
        return response, token

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> OpenAIChatStream:
        """Stream a chat conversation, yielding text chunks.

        Usage and tool_calls are available on the returned stream object after iteration.
        """
        return OpenAIChatStream(self, list(messages), list(tools) if tools else None)


@dataclasses.dataclass
class OpenAIChatStream(ChatStream, OpenAIToolMixin):
    """ChatStream implementation for OpenAI-compatible APIs."""

    _client: _OpenAIBase
    _messages: list[ChatMessage]
    _tools: list[ToolDefinition] | None = None
    usage: UsageToken | None = None
    tool_calls: list[ToolCall] | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        if not self._client.model:
            raise LLMError(self._client.provider, "No model specified")
        body: CreateChatCompletionRequest = {
            "model": self._client.model,
            "messages": cast(list, self._messages),
            "temperature": self._client.temperature,
            "stream_options": {"include_usage": True},
        }
        if self._tools:
            body["tools"] = cast(list, self._tools_to_openai(self._tools))

        # Accumulate tool call deltas keyed by index
        pending: dict[int, dict[str, str]] = {}

        async for chunk in self._client.stream(body):
            if text := self._extract_text(chunk):
                yield text
            self._accumulate_tool_call_deltas(chunk, pending)
            if usage := self._extract_usage(chunk):
                self.usage = usage

        if pending:
            self.tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type="function",
                    function=ToolCallFunction(
                        name=tc["name"], arguments=tc["arguments"]
                    ),
                )
                for tc in (pending[i] for i in sorted(pending))
            ]

    def _extract_text(self, chunk: CreateChatCompletionStreamResponse) -> str | None:
        """Extract text content from an OpenAI stream response chunk."""
        if choices := chunk.get("choices"):
            if delta := choices[0].get("delta"):
                return delta.get("content")
        return None

    def _extract_usage(
        self, chunk: CreateChatCompletionStreamResponse
    ) -> UsageToken | None:
        """Extract usage info from an OpenAI stream response chunk (final chunk)."""
        if usage := chunk.get("usage"):
            token: UsageToken = {
                "total": usage.get("total_tokens", 0),
                "input": usage.get("prompt_tokens", 0),
                "output": usage.get("completion_tokens", 0),
            }
            if details := usage.get("prompt_tokens_details"):
                if (cached := details.get("cached_tokens")) is not None:
                    token["cached"] = cached
            return token
        return None


@dataclasses.dataclass
class OpenAIClient(_OpenAIBase):
    """OpenAI API client with chat and batch support."""

    async def upload_batch_file(
        self,
        requests: list[BatchRequest],
        model: str | None = None,
    ) -> str:
        """Serialize batch requests to JSONL and upload via POST /files.

        Each line in the JSONL file is an OpenAI batch request object with
        `custom_id`, `method`, `url`, and `body` fields. Returns the file ID.
        """
        _model = model or self.model
        if not _model:
            raise LLMError(self.provider, "No model specified")
        lines: list[str] = []
        for idx, req in enumerate(requests):
            line = {
                "custom_id": req.custom_id or f"request-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {**req.body, "model": _model},
            }
            lines.append(json.dumps(line))
        content = "\n".join(lines)

        resp = await self.session.post(
            "/files",
            data={"purpose": "batch"},
            files={"file": ("batch.jsonl", content, "application/jsonl")},
        )
        data: OpenAIFile = _check_resp(resp)
        return data["id"]

    async def create_batch(
        self,
        requests: list[BatchRequest],
        model: str | None = None,
        metadata: dict[str, str] | None = None,
        completion_window: str = "24h",
    ) -> BatchJob:
        """Create an OpenAI batch from a list of requests.

        Uploads the requests as a JSONL file, then creates a batch
        referencing that file. The `completion_window` defaults to "24h"
        (currently the only value OpenAI supports).
        """
        file_id = await self.upload_batch_file(requests, model)
        payload: dict = {
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": completion_window,
        }
        if metadata:
            payload["metadata"] = metadata
        resp = await self.session.post("/batches", json=payload)
        data: Batch = _check_resp(resp)
        return BatchJob.load(data)

    async def get_batch(self, batch_id: str) -> BatchJob:
        """Get the current status of a batch."""
        resp = await self.session.get(f"/batches/{batch_id}")
        data: Batch = _check_resp(resp)
        return BatchJob.load(data)

    async def list_batches(
        self,
        limit: int = 20,
        after: str | None = None,
    ) -> tuple[list[BatchJob], str | None]:
        """List batches with cursor-based pagination.

        Returns a tuple of (jobs, next_cursor). Pass `next_cursor` as `after`
        to fetch the next page.
        """
        params: dict[str, str] = {"limit": str(limit)}
        if after:
            params["after"] = after
        resp = await self.session.get("/batches", params=params)
        data: ListBatchesResponse = _check_resp(resp)
        jobs = [BatchJob.load(item) for item in data.get("data", [])]
        next_cursor: str | None = None
        if data.get("has_more"):
            next_cursor = data.get("last_id")
        return jobs, next_cursor

    async def cancel_batch(self, batch_id: str) -> BatchJob:
        """Cancel a batch that is in progress."""
        resp = await self.session.post(f"/batches/{batch_id}/cancel")
        data: Batch = _check_resp(resp)
        return BatchJob.load(data)

    async def get_batch_results(self, output_file_id: str) -> list[BatchResult]:
        """Download and parse the output JSONL file for a completed batch."""
        resp = await self.session.get(f"/files/{output_file_id}/content")
        _check_resp_status(resp)
        text = resp.text or ""
        results: list[BatchResult] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            results.append(BatchResult.from_line(json.loads(line)))
        return results
