from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import TYPE_CHECKING, ClassVar, Literal, cast, get_args

import niquests
from urllib3.util.retry import Retry

from .._base import ChatStream, LLMClientBase, LLMError, Provider
from ..conversation import Message
from ..errors import QuotaExceededError, TooManyRequestsError
from ..models import UsageToken

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
    "OpenAICompatibleClient",
    "OpenAIClient",
    "OPENAI_MODELS",
    "OPENAI_ENDPOINT",
    "OpenAIModel",
    "is_openai_model",
)


def _check_resp(resp: niquests.Response) -> dict:
    """Check OpenAI-compatible HTTP response, extracting error details."""
    try:
        resp.raise_for_status()
    except niquests.exceptions.HTTPError as e:
        data = resp.json()
        if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            retry_after = resp.headers.get("retry-after")
            retry_delay = int(retry_after) if retry_after else 60
            raise TooManyRequestsError(retry_delay=retry_delay, response=resp)
        if resp.status_code == HTTPStatus.PAYMENT_REQUIRED:
            raise QuotaExceededError(body=data)
        raise e
    return resp.json()


_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def is_openai_model(model_name: str | None) -> bool:
    """Check if the model name looks like an OpenAI model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith(_OPENAI_PREFIXES)


OPENAI_MODELS: set[str] = set(get_args(OpenAIModel))

OPENAI_ENDPOINT = "https://api.openai.com/v1/"


@dataclasses.dataclass
class OpenAICompatibleClient(LLMClientBase[Retry]):
    """Base client for OpenAI-compatible APIs.

    Can be used directly for arbitrary OpenAI-compatible endpoints (Ollama,
    Together AI, vLLM, etc.) or subclassed by named providers like OpenAIClient,
    MistralClient, and GrokClient.
    """

    provider: ClassVar[Provider] = "openai_compat"
    base_url: str = ""
    _retry: Retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"],
    )

    def _get_default_api_key(self) -> str:
        raise LLMError(self.provider, "api_key is required")

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
        data = cast("CreateChatCompletionResponse", _check_resp(resp))

        usage: CompletionUsage | None = data.get("usage")
        if usage is None:
            raise LLMError(self.provider, "No usage in response")
        token: UsageToken = {
            "total": usage["total_tokens"],
            "input": usage["prompt_tokens"],
            "output": usage["completion_tokens"],
        }
        if details := usage.get("prompt_tokens_details"):
            if cached := details.get("cached_tokens"):
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
        resp.raise_for_status()

        async for line in resp.iter_lines(decode_unicode=True):  # pyright: ignore[reportGeneralTypeIssues]
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                yield cast("CreateChatCompletionStreamResponse", json.loads(data_str))

    async def complete_chat(self, messages: list[Message]) -> tuple[str, UsageToken]:
        """Send a chat conversation and return the complete response."""
        body: CreateChatCompletionRequest = {
            "model": self.model or "",
            "messages": cast(list, messages),
            "temperature": self.temperature,
        }
        data, token = await self.complete(body)
        text = data["choices"][0]["message"]["content"] or ""
        return text, token

    def stream_chat(self, messages: list[Message]) -> OpenAIChatStream:
        """Stream a chat conversation, yielding text chunks.

        Usage is available on the returned stream object after iteration.
        """
        return OpenAIChatStream(self, messages)


@dataclasses.dataclass
class OpenAIClient(OpenAICompatibleClient):
    """OpenAI API client."""

    provider: ClassVar[Provider] = "openai"
    model: str | None = "gpt-4o"
    base_url: str = OPENAI_ENDPOINT

    def _get_default_api_key(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMError(self.provider, "OPENAI_API_KEY not set")
        return api_key


class OpenAIChatStream(ChatStream):
    """ChatStream implementation for OpenAI-compatible APIs."""

    def __init__(self, client: OpenAICompatibleClient, messages: list[Message]) -> None:
        self._client = client
        self._messages = messages
        self.usage: UsageToken | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        body: CreateChatCompletionRequest = {
            "model": self._client.model or "",
            "messages": cast(list, self._messages),
            "temperature": self._client.temperature,
            "stream_options": {"include_usage": True},
        }

        async for chunk in self._client.stream(body):
            if text := self._extract_text(chunk):
                yield text
            if usage := self._extract_usage(chunk):
                self.usage = usage

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
                if cached := details.get("cached_tokens"):
                    token["cached"] = cached
            return token
        return None
