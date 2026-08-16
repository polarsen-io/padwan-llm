import json
from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock

import pytest

from padwan_llm.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_llm.openai.batch import BatchJob, BatchResult
from padwan_llm.openai.client import (
    OpenAIChatStream,
    OpenAIClient,
    _check_resp,
    _extract_text_payload,
    _extract_thought_payload,
)
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
    stream free of any thought content. When `on_thought` is unset,
    reasoning chunks are silently dropped — never raised, never yielded.
    """

    @pytest.mark.parametrize(
        "chunks, expected_text, expected_thoughts, with_callback",
        [
            pytest.param(
                [
                    {"choices": [{"delta": {"reasoning_content": "Let me think... "}}]},
                    {"choices": [{"delta": {"reasoning_content": "ok done."}}]},
                    {"choices": [{"delta": {"content": "The answer is 42"}}]},
                ],
                "The answer is 42",
                ["Let me think... ", "ok done."],
                True,
                id="grok_reasoning_content",
            ),
            pytest.param(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": [
                                        {
                                            "type": "thinking",
                                            "thinking": [
                                                {
                                                    "type": "text",
                                                    "text": "Reasoning step 1",
                                                }
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
                ],
                "Final answer.",
                ["Reasoning step 1"],
                True,
                id="mistral_thinking_chunks",
            ),
            pytest.param(
                [
                    {
                        "choices": [
                            {"delta": {"reasoning_content": "internal scratchpad"}}
                        ]
                    },
                    {"choices": [{"delta": {"content": "answer"}}]},
                ],
                "answer",
                [],
                False,  # on_thought=None
                id="no_callback_drops_silently",
            ),
        ],
    )
    async def test_forwards_or_drops_thoughts(
        self,
        chunks: list[dict],
        expected_text: str,
        expected_thoughts: list[str],
        with_callback: bool,
    ):
        async def fake_stream(_body):
            for c in chunks:
                yield c

        thoughts: list[str] = []
        client = MagicMock(spec=OpenAIClient)
        client.model = "test-model"
        client.temperature = 0.2
        client.provider = "openai"
        client.on_thought = thoughts.append if with_callback else None
        client.stream = fake_stream
        stream = OpenAIChatStream(client, [])
        produced = [t async for t in stream]
        assert "".join(produced) == expected_text
        assert thoughts == expected_thoughts


class TestOpenAIStream:
    """Tests for _OpenAIBase.stream() SSE parsing."""

    @pytest.fixture
    def client(self):
        c = MagicMock(spec=OpenAIClient)
        c.provider = "openai"
        c.model = "gpt-4o"
        c.base_url = "https://api.openai.com/v1/"
        c._sse_url = OpenAIClient._sse_url.__get__(c)
        c.session = AsyncMock()
        return c

    @pytest.mark.asyncio
    async def test_yields_parsed_chunks(self, client, make_sse_event, make_sse_resp):
        chunk1 = {"choices": [{"delta": {"content": "Hello"}}]}
        chunk2 = {"choices": [{"delta": {"content": " world"}}]}
        events = [
            make_sse_event(json.dumps(chunk1)),
            make_sse_event(json.dumps(chunk2)),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk1, chunk2]

    @pytest.mark.asyncio
    async def test_done_sentinel_stops(self, client, make_sse_event, make_sse_resp):
        chunk = {"choices": [{"delta": {"content": "hi"}}]}
        done_ev = MagicMock()
        done_ev.data = "[DONE]"
        done_ev.json = MagicMock(side_effect=ValueError)
        events = [
            make_sse_event(json.dumps(chunk)),
            done_ev,
            make_sse_event(json.dumps({"should": "not appear"})),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk]

    @pytest.mark.asyncio
    async def test_skips_keepalive_frames(self, client, make_sse_event, make_sse_resp):
        chunk = {"choices": [{"delta": {"content": "ok"}}]}
        events = [
            make_sse_event(""),
            make_sse_event(json.dumps(chunk)),
            make_sse_event(""),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [
            c
            async for c in OpenAIClient.stream(
                client, {"model": "gpt-4o", "messages": []}
            )
        ]
        assert chunks == [chunk]

    @pytest.mark.asyncio
    async def test_no_extension_raises(self, client):
        resp = AsyncMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock(return_value=resp)
        resp.extension = None
        client.session.post.return_value = resp
        with pytest.raises(LLMError, match="SSE extension"):
            _ = [
                c
                async for c in OpenAIClient.stream(
                    client, {"model": "gpt-4o", "messages": []}
                )
            ]

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self, client, make_sse_resp):
        ev = MagicMock()
        ev.data = "{bad json"
        ev.json = MagicMock(side_effect=ValueError("parse error"))
        client.session.post.return_value = make_sse_resp([ev])
        with pytest.raises(LLMError, match="Stream parse error"):
            _ = [
                c
                async for c in OpenAIClient.stream(
                    client, {"model": "gpt-4o", "messages": []}
                )
            ]


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
