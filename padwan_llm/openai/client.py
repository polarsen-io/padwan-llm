import dataclasses
from dataclasses import field
from functools import partial
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
    "OpenAIClient",
    "OPENAI_MODELS",
    "OPENAI_ENDPOINT",
    "OpenAIModel",
    "is_openai_model",
)


async def _check_resp_status(resp: niquests.Response | niquests.AsyncResponse) -> None:
    """Check HTTP status and raise appropriate errors without consuming the body."""
    try:
        resp.raise_for_status()
    except niquests.exceptions.HTTPError as e:
        try:
            data = await resp.json()
        except Exception:
            raise e
        if resp.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            retry_after = resp.headers.get("retry-after")
            retry_delay = int(retry_after) if retry_after else 60
            raise TooManyRequestsError(retry_delay=retry_delay, response=resp)
        if resp.status_code == HTTPStatus.PAYMENT_REQUIRED:
            raise QuotaExceededError(body=data)
        raise e


async def _check_resp(resp: niquests.Response | niquests.AsyncResponse) -> dict:
    """Check OpenAI-compatible HTTP response and return the parsed JSON body."""
    await _check_resp_status(resp)
    return await resp.json()


_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4", "chatgpt-")


def is_openai_model(model_name: str | None) -> bool:
    """Check if the model name looks like an OpenAI model based on known prefixes."""
    if model_name is None:
        return False
    return model_name.startswith(_OPENAI_PREFIXES)


OPENAI_MODELS: set[str] = set(get_args(OpenAIModel))

OPENAI_ENDPOINT = "https://api.openai.com/v1/"


@dataclasses.dataclass
class OpenAIClient(LLMClientBase[Retry]):
    """OpenAI API client.

    Also serves as the base for other OpenAI-compatible providers (Grok,
    Mistral, etc.) since they share the same protocol.
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
        data = cast("CreateChatCompletionResponse", await _check_resp(resp))

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
        await _check_resp_status(resp)

        async for line in resp.iter_lines(decode_unicode=True):  # pyright: ignore[reportGeneralTypeIssues]
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]  # Remove "data: " prefix
                if data_str == "[DONE]":
                    break
                try:
                    yield cast(
                        "CreateChatCompletionStreamResponse", json.loads(data_str)
                    )
                except json.JSONDecodeError as e:
                    raise LLMError(self.provider, f"Stream parse error: {e}") from e

    async def complete_chat(self, messages: list[Message]) -> tuple[str, UsageToken]:
        """Send a chat conversation and return the complete response."""
        if not self.model:
            raise LLMError(self.provider, "No model specified")
        body: CreateChatCompletionRequest = {
            "model": self.model,
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


class OpenAIChatStream(ChatStream):
    """ChatStream implementation for OpenAI-compatible APIs."""

    def __init__(self, client: OpenAIClient, messages: list[Message]) -> None:
        self._client = client
        self._messages = messages
        self.usage: UsageToken | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        if not self._client.model:
            raise LLMError(self._client.provider, "No model specified")
        body: CreateChatCompletionRequest = {
            "model": self._client.model,
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
                if (cached := details.get("cached_tokens")) is not None:
                    token["cached"] = cached
            return token
        return None
