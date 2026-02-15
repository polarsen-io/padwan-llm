from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
from niquests.exceptions import HTTPError

from typing import TYPE_CHECKING

from padwan_llm.errors import QuotaExceededError, TooManyRequestsError
from padwan_llm.openai.client import OpenAIClient, OpenAIChatStream, _check_resp

if TYPE_CHECKING:
    from padwan_llm.openai.types import CreateChatCompletionStreamResponse


@pytest.mark.parametrize(
    "status, json_data, headers, ctx",
    [
        pytest.param(200, {"ok": True}, None, nullcontext(), id="success"),
        pytest.param(
            429,
            {},
            {"retry-after": "30"},
            pytest.raises(TooManyRequestsError),
            id="429-retry",
        ),
        pytest.param(
            429, {}, None, pytest.raises(TooManyRequestsError), id="429-default-60"
        ),
        pytest.param(
            402,
            {"error": "quota"},
            None,
            pytest.raises(QuotaExceededError),
            id="402-quota",
        ),
        pytest.param(
            500, {"error": "server"}, None, pytest.raises(HTTPError), id="500"
        ),
    ],
)
async def test_check_resp(status, json_data, headers, ctx, make_resp):
    resp = make_resp(status, json_data, headers)
    with ctx:
        result = await _check_resp(resp)
        assert result == json_data


class TestOpenAIChatStreamExtraction:
    def setup_method(self):
        client = MagicMock(spec=OpenAIClient)
        self.stream = OpenAIChatStream(client, [])

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            pytest.param(
                {"choices": [{"delta": {"content": "hello"}}]}, "hello", id="text"
            ),
            pytest.param({"choices": [{"delta": {}}]}, None, id="empty-delta"),
            pytest.param({}, None, id="empty"),
        ],
    )
    def test_extract_text(
        self, chunk: CreateChatCompletionStreamResponse, expected: str | None
    ):
        assert self.stream._extract_text(chunk) == expected

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            pytest.param(
                {
                    "usage": {
                        "total_tokens": 100,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                    }
                },
                {"total": 100, "input": 80, "output": 20},
                id="basic",
            ),
            pytest.param(
                {
                    "usage": {
                        "total_tokens": 100,
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 10},
                    }
                },
                {"total": 100, "input": 80, "output": 20, "cached": 10},
                id="cached",
            ),
            pytest.param({}, None, id="no-usage"),
        ],
    )
    def test_extract_usage(
        self, chunk: CreateChatCompletionStreamResponse, expected: dict | None
    ):
        assert self.stream._extract_usage(chunk) == expected
