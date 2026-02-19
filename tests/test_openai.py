from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from typing import TYPE_CHECKING

from padwan_llm.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_llm.openai.batch import BatchJob, BatchResult
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
        pytest.param(500, {"error": "server"}, None, pytest.raises(LLMError), id="500"),
    ],
)
def test_check_resp(status, json_data, headers, ctx, make_resp):
    resp = make_resp(status, json_data, headers)
    with ctx:
        result = _check_resp(resp)
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


class TestBatchResultFromLine:
    @pytest.mark.parametrize(
        "data, expected",
        [
            pytest.param(
                {
                    "custom_id": "req-1",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": "Hello!"}, "index": 0}],
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 5,
                                "total_tokens": 15,
                            },
                        },
                    },
                },
                BatchResult(
                    custom_id="req-1",
                    content="Hello!",
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                ),
                id="full-response",
            ),
            pytest.param(
                {
                    "custom_id": "req-2",
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [], "usage": {}},
                    },
                },
                BatchResult(custom_id="req-2", content=""),
                id="empty-choices",
            ),
            pytest.param(
                {
                    "custom_id": "req-3",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [{"message": {"content": None}, "index": 0}],
                            "usage": {"prompt_tokens": 5, "total_tokens": 5},
                        },
                    },
                },
                BatchResult(
                    custom_id="req-3",
                    content="",
                    input_tokens=5,
                    total_tokens=5,
                ),
                id="null-content",
            ),
            pytest.param(
                {"custom_id": "req-4", "error": {"code": "server_error"}},
                BatchResult(custom_id="req-4", content=""),
                id="error-response",
            ),
        ],
    )
    def test_from_line(self, data: dict, expected: BatchResult):
        assert BatchResult.from_line(data) == expected


class TestBatchJobProperties:
    @pytest.mark.parametrize(
        "status, expected",
        [
            pytest.param("completed", True, id="completed"),
            pytest.param("failed", True, id="failed"),
            pytest.param("expired", True, id="expired"),
            pytest.param("cancelled", True, id="cancelled"),
            pytest.param("in_progress", False, id="in_progress"),
            pytest.param("validating", False, id="validating"),
            pytest.param("finalizing", False, id="finalizing"),
            pytest.param("cancelling", False, id="cancelling"),
        ],
    )
    def test_is_terminal(self, status: str, expected: bool):
        job = BatchJob(
            id="batch_123",
            status=status,  # pyright: ignore[reportArgumentType]
            endpoint="/v1/chat/completions",
            input_file_id="file-abc",
        )
        assert job.is_terminal is expected

    @pytest.mark.parametrize(
        "status, expected",
        [
            pytest.param("completed", True, id="completed"),
            pytest.param("failed", False, id="failed"),
            pytest.param("in_progress", False, id="in_progress"),
        ],
    )
    def test_succeeded(self, status: str, expected: bool):
        job = BatchJob(
            id="batch_123",
            status=status,  # pyright: ignore[reportArgumentType]
            endpoint="/v1/chat/completions",
            input_file_id="file-abc",
        )
        assert job.succeeded is expected
