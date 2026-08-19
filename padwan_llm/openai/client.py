from __future__ import annotations

import dataclasses
import typing
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import field
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar, Literal, cast, get_args

import niquests
from urllib3.util.retry import Retry

from .._base import ChatStream, LLMClientBase, Provider, env_api_key
from .._json import dumps as _json_dumps, loads as _json_loads
from ..conversation import ChatMessage
from ..errors import LLMError, QuotaExceededError, TooManyRequestsError
from ..models import (
    ChatResponse,
    FinishReason,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    UsageToken,
)
from .batch import BatchJob, BatchRequest, BatchResult
from .tools import OpenAIToolMixin
from .types import Batch, ListBatchesResponse, OpenAIFile

# Normalize provider-specific finish reasons to our FinishReason values.
# https://platform.openai.com/docs/api-reference/chat/object#chat/object-choices
_OPENAI_FINISH_REASON_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "tool_calls",
    "model_length": "length",
    "error": "error",
}


def _extract_text_payload(payload: dict[str, typing.Any]) -> str | None:
    """Pull plain answer text out of a `delta` or `message` dict.

    Handles two shapes interchangeably: a plain string `content` (OpenAI,
    Grok), or Mistral's structured `content: list[ContentChunk]` where
    we keep only `type: "text"` chunks and concatenate them. `thinking`
    chunks are ignored here — they go through `_extract_thought_payload`.
    """
    content = payload.get("content")
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        texts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                if t := chunk.get("text"):
                    texts.append(t)
        if texts:
            return "".join(texts)
    return None


def _extract_thought_payload(payload: dict[str, typing.Any]) -> str | None:
    """Pull reasoning text out of a `delta` or `message` dict.

    Handles two shapes:

    - `reasoning_content: str` — xAI/Grok and DeepSeek-style reasoning
      models surface their scratchpad as a sibling string field on the
      delta/message.
    - `content: list[{"type": "thinking", "thinking": [{"type": "text",
      "text": ...}]}]` — Mistral's `Magistral` models embed reasoning as
      a `ThinkChunk` inside a structured content array.

    Returns the concatenated thought text, or `None` if neither shape
    carries any reasoning content.
    """
    if rc := payload.get("reasoning_content"):
        return rc if isinstance(rc, str) else None
    content = payload.get("content")
    if isinstance(content, list):
        thoughts: list[str] = []
        for chunk in content:
            if not isinstance(chunk, dict) or chunk.get("type") != "thinking":
                continue
            for inner in chunk.get("thinking") or []:
                if isinstance(inner, dict) and inner.get("type") == "text":
                    if t := inner.get("text"):
                        thoughts.append(t)
        if thoughts:
            return "".join(thoughts)
    return None


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
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.1-mini",
    "gpt-5.1-codex",
    "gpt-5.2",
    "gpt-5.2-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3",
    "o3-mini",
    "o4-mini",
    "codex-mini-latest",
]

__all__ = (
    "OPENAI_CHAT_MODELS",
    "OPENAI_ENDPOINT",
    "OPENAI_MODELS",
    "OpenAIClient",
    "OpenAIModel",
    "_OpenAIBase",
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


def _check_resp[T](
    resp: niquests.Response, *, decoder: Callable[[bytes], T] | None = None
) -> T:
    """Check OpenAI-compatible HTTP response and return the parsed JSON body,
    deserialized through `decoder` when provided (e.g. msgspec for typed validation)."""
    checked = _check_resp_status(resp)
    if decoder is None:
        return checked.json()
    if not (body := checked.content):
        raise LLMError("openai", "Empty response body")
    return decoder(body)


_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-", "codex-")


def is_openai_model(model_name: str | None) -> bool:
    """Check if the model name looks like an OpenAI model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith(_OPENAI_PREFIXES)


OPENAI_MODELS: set[str] = set(get_args(OpenAIModel))

# Models in OPENAI_MODELS that do not accept /v1/chat/completions requests:
#   - gpt-5.2-pro: reasoning/pro tier, exposed only via the /v1/responses API
#   - codex-mini-latest: legacy /v1/completions endpoint (text completion)
_OPENAI_NON_CHAT_MODELS: frozenset[str] = frozenset(
    {"gpt-5.2-pro", "codex-mini-latest"}
)

OPENAI_CHAT_MODELS: frozenset[str] = frozenset(OPENAI_MODELS - _OPENAI_NON_CHAT_MODELS)

OPENAI_ENDPOINT = "https://api.openai.com/v1/"


class _OpenAIAuth:
    """OpenAI provider identity, shared by the chat and realtime clients."""

    provider: ClassVar[Provider] = "openai"

    def _get_default_api_key(self) -> str:
        return env_api_key(self.provider, "OPENAI_API_KEY")


@dataclasses.dataclass
class _OpenAIBase(_OpenAIAuth, LLMClientBase[Retry], OpenAIToolMixin):
    """OpenAI-compatible base client with chat and auth logic.

    Shared by OpenAIClient, GrokClient, and MistralClient. Does not include
    batch methods — those live on the concrete provider clients.
    """

    model: str | None = "gpt-4o"
    base_url: str = OPENAI_ENDPOINT
    _retry: Retry = field(
        default_factory=partial(
            Retry,
            total=3,
            backoff_factor=0.5,
            status_forcelist=[
                # Standard
                500,  # Internal Server Error
                502,  # Bad Gateway
                503,  # Service Unavailable
                504,  # Gateway Timeout
                # Cloudflare-proprietary
                520,  # Unknown Error
                522,  # Connection Timed Out
                524,  # A Timeout Occurred
            ],
            allowed_methods=["POST"],
        )
    )

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
            self._sse_url("/chat/completions"),
            json={**body, "stream": True},
        )
        _check_resp_status(resp)
        ext = resp.extension
        if ext is None:
            raise LLMError(self.provider, "SSE extension not available on response")
        while not ext.closed:
            event = await ext.next_payload()
            if event is None:
                break
            if not event.data:
                continue
            if event.data == "[DONE]":
                break
            try:
                yield cast("CreateChatCompletionStreamResponse", event.json())
            except ValueError as e:
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
        message_dict = cast(dict[str, typing.Any], choice["message"])
        raw_reason = choice.get("finish_reason", "stop")
        # Forward any reasoning content to on_thought before unpacking the
        # answer text — keeps the final ChatResponse free of the model's
        # internal scratchpad. Same shape on both wire formats handled by
        # `_extract_thought_payload` (Grok `reasoning_content`, Mistral
        # `ThinkChunk` inside a structured content array).
        if self.on_thought and (thought := _extract_thought_payload(message_dict)):
            self.on_thought(thought)
        response: ChatResponse = {
            "content": _extract_text_payload(message_dict),
            "finish_reason": _OPENAI_FINISH_REASON_MAP.get(raw_reason, "other"),
        }
        if raw_tool_calls := message_dict.get("tool_calls"):
            response["tool_calls"] = self._extract_tool_calls(
                cast(list, raw_tool_calls)
            )
        return response, token

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
        extra_params: dict[str, typing.Any] | None = None,
    ) -> OpenAIChatStream:
        """Stream a chat conversation, yielding text chunks.

        Usage and tool_calls are available on the returned stream object after iteration.
        The optional ``extra_params`` dict is merged into the request body verbatim,
        so provider-specific fields (e.g. NVIDIA's ``chat_template_kwargs``) can be
        passed without modifying the client.
        """
        return OpenAIChatStream(self, messages, tools, extra_params=extra_params)


@dataclasses.dataclass
class OpenAIChatStream(ChatStream, OpenAIToolMixin):
    """ChatStream implementation for OpenAI-compatible APIs."""

    client: _OpenAIBase
    messages: Sequence[ChatMessage]
    tools: Sequence[ToolDefinition] | None = None
    extra_params: dict[str, typing.Any] | None = None
    usage: UsageToken | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        if not self.client.model:
            raise LLMError(self.client.provider, "No model specified")
        body: CreateChatCompletionRequest = {
            "model": self.client.model,
            "messages": cast(list, self.messages),
            "temperature": self.client.temperature,
            "stream_options": {"include_usage": True},
        }
        if self.tools:
            body["tools"] = cast(list, self._tools_to_openai(self.tools))
        if self.extra_params:
            body.update(cast(typing.Any, self.extra_params))

        # Accumulate tool call deltas keyed by index
        pending: dict[int, dict[str, str]] = {}

        async for chunk in self.client.stream(body):
            delta = self._delta(chunk)
            if delta is not None:
                if self.client.on_thought and (
                    thought := _extract_thought_payload(delta)
                ):
                    self.client.on_thought(thought)
                if text := _extract_text_payload(delta):
                    yield text
            if choices := chunk.get("choices"):
                if reason := choices[0].get("finish_reason"):
                    self.finish_reason = reason
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

    @staticmethod
    def _delta(
        chunk: CreateChatCompletionStreamResponse,
    ) -> dict[str, typing.Any] | None:
        """Return the first choice's `delta` dict, or None if absent."""
        if choices := chunk.get("choices"):
            if delta := choices[0].get("delta"):
                return cast(dict[str, typing.Any], delta)
        return None

    def _extract_text(self, chunk: CreateChatCompletionStreamResponse) -> str | None:
        """Extract text content from an OpenAI stream response chunk.

        Kept for backwards compatibility with subclasses that still call
        it. New code should go through `__aiter__`'s shared payload
        helpers (`_extract_text_payload`, `_extract_thought_payload`).
        """
        delta = self._delta(chunk)
        return _extract_text_payload(delta) if delta else None

    @staticmethod
    def _extract_usage(chunk: CreateChatCompletionStreamResponse) -> UsageToken | None:
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
            lines.append(_json_dumps(line))
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
            results.append(BatchResult.from_line(_json_loads(line)))
        return results
