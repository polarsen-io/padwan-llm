from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest

from typing import TYPE_CHECKING

from padwan_llm.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_llm.openai.batch import BatchJob, BatchResult
from padwan_llm.openai.client import (
    OpenAIClient,
    OpenAIChatStream,
    _check_resp,
    _extract_text_payload,
    _extract_thought_payload,
)

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
            pytest.param(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": [
                                    {"type": "text", "text": "Hello "},
                                    {"type": "text", "text": "world"},
                                ]
                            }
                        }
                    ]
                },
                "Hello world",
                id="mistral-text-chunks",
            ),
            pytest.param(
                {
                    "choices": [
                        {
                            "delta": {
                                "content": [
                                    {
                                        "type": "thinking",
                                        "thinking": [
                                            {"type": "text", "text": "Thinking..."}
                                        ],
                                    }
                                ]
                            }
                        }
                    ]
                },
                None,
                id="mistral-thinking-only",
            ),
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


class TestPayloadHelpers:
    """`_extract_text_payload` and `_extract_thought_payload` are the
    cross-provider routers used by `OpenAIChatStream` and
    `_OpenAIBase.complete_chat`. They have to handle three wire shapes:
    OpenAI/Grok plain string, Mistral structured chunk arrays, and the
    Grok/DeepSeek `reasoning_content` sibling field.
    """

    @pytest.mark.parametrize(
        "payload, expected",
        [
            pytest.param({"content": "hi"}, "hi", id="plain-string"),
            pytest.param({"content": ""}, None, id="empty-string"),
            pytest.param({"content": None}, None, id="null"),
            pytest.param({}, None, id="missing"),
            pytest.param(
                {
                    "content": [
                        {"type": "text", "text": "Hello "},
                        {"type": "text", "text": "world"},
                    ]
                },
                "Hello world",
                id="mistral-text-chunks",
            ),
            pytest.param(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "ignore me"}],
                        },
                        {"type": "text", "text": "the answer"},
                    ]
                },
                "the answer",
                id="mistral-mixed-skips-thinking",
            ),
            pytest.param(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "only thinking"}],
                        }
                    ]
                },
                None,
                id="mistral-thinking-only",
            ),
        ],
    )
    def test_extract_text_payload(self, payload: dict, expected: str | None):
        assert _extract_text_payload(payload) == expected

    @pytest.mark.parametrize(
        "payload, expected",
        [
            pytest.param(
                {"reasoning_content": "thinking..."}, "thinking...", id="grok"
            ),
            pytest.param({"reasoning_content": ""}, None, id="grok-empty"),
            pytest.param({"content": "just text"}, None, id="plain-no-thought"),
            pytest.param(
                {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [
                                {"type": "text", "text": "first "},
                                {"type": "text", "text": "second"},
                            ],
                        },
                        {"type": "text", "text": "answer"},
                    ]
                },
                "first second",
                id="mistral-thinking",
            ),
            pytest.param(
                {
                    "content": [
                        {"type": "text", "text": "no thinking here"},
                    ]
                },
                None,
                id="mistral-text-only",
            ),
            pytest.param({}, None, id="missing"),
        ],
    )
    def test_extract_thought_payload(self, payload: dict, expected: str | None):
        assert _extract_thought_payload(payload) == expected


class TestStreamForwardsThoughts:
    """`OpenAIChatStream.__aiter__` must forward reasoning chunks to
    `client.on_thought` for both Grok-style `reasoning_content` and
    Mistral-style `ThinkChunk` payloads, while keeping the regular text
    stream free of any thought content.
    """

    async def _drive(
        self,
        chunks: list[dict],
        on_thought: list[str],
    ) -> tuple[list[str], list[str]]:
        """Run a fake stream through `OpenAIChatStream` and return
        (yielded_text_chunks, on_thought_calls)."""

        async def fake_stream(_body):
            for c in chunks:
                yield c

        client = MagicMock(spec=OpenAIClient)
        client.model = "test-model"
        client.temperature = 0.2
        client.provider = "openai"
        client.on_thought = on_thought.append
        client.stream = fake_stream
        stream = OpenAIChatStream(client, [])
        produced = [t async for t in stream]
        return produced, on_thought

    async def test_grok_reasoning_content(self):
        """xAI/Grok surfaces reasoning via `delta.reasoning_content`."""
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "Let me think... "}}]},
            {"choices": [{"delta": {"reasoning_content": "ok done."}}]},
            {"choices": [{"delta": {"content": "The answer is 42"}}]},
        ]
        thoughts: list[str] = []
        text_chunks, _ = await self._drive(chunks, thoughts)
        assert "".join(text_chunks) == "The answer is 42"
        assert thoughts == ["Let me think... ", "ok done."]

    async def test_mistral_thinking_chunks(self):
        """Mistral Magistral surfaces reasoning via structured `ThinkChunk`."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": [
                                        {"type": "text", "text": "Reasoning step 1"}
                                    ],
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "text", "text": "Final answer."},
                            ]
                        }
                    }
                ]
            },
        ]
        thoughts: list[str] = []
        text_chunks, _ = await self._drive(chunks, thoughts)
        assert "".join(text_chunks) == "Final answer."
        assert thoughts == ["Reasoning step 1"]

    async def test_no_callback_when_unset(self):
        """If `on_thought` is None, reasoning content is silently dropped
        from the text stream — never raised, never yielded."""

        async def fake_stream(_body):
            yield {"choices": [{"delta": {"reasoning_content": "internal scratchpad"}}]}
            yield {"choices": [{"delta": {"content": "answer"}}]}

        client = MagicMock(spec=OpenAIClient)
        client.model = "test-model"
        client.temperature = 0.2
        client.provider = "openai"
        client.on_thought = None
        client.stream = fake_stream
        stream = OpenAIChatStream(client, [])
        produced = [t async for t in stream]
        assert "".join(produced) == "answer"


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
