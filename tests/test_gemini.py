import json
import re
from contextlib import nullcontext
from typing import get_type_hints
from unittest.mock import AsyncMock, MagicMock

import pytest
from google.genai.types import (
    ContentDict,
    GenerationConfigDict,
    PartDict,
    ThinkingConfigDict,
)

from padwan_llm.conversation import Message
from padwan_llm.errors import LLMError, QuotaExceededError, TooManyRequestsError
from padwan_llm.gemini.batch import BatchJob, BatchResult
from padwan_llm.gemini.models import (
    BatchState,
    Content,
    GenerationConfig,
    InlinedResponse,
    Part,
    ThinkingConfig,
)
from padwan_llm.gemini.client import (
    GeminiClient,
    GeminiChatStream,
    _check_resp,
    _parse_retry_delay,
)


@pytest.mark.parametrize(
    "delay_str, expected",
    [
        pytest.param("48s", 48, id="integer"),
        pytest.param("48.249s", 49, id="fractional-rounds-up"),
        pytest.param("1.1s", 2, id="small-fractional"),
    ],
)
def test_parse_retry_delay(delay_str: str, expected: int):
    assert _parse_retry_delay(delay_str) == expected


@pytest.mark.parametrize(
    "status, json_data, ctx",
    [
        pytest.param(200, {"candidates": []}, nullcontext(), id="success"),
        pytest.param(
            429,
            {
                "error": {
                    "details": [
                        {"@type": "type.googleapis.com/google.rpc.QuotaFailure"}
                    ]
                }
            },
            pytest.raises(QuotaExceededError),
            id="429-quota",
        ),
        pytest.param(
            429,
            {
                "error": {
                    "details": [
                        {
                            "@type": "type.googleapis.com/google.rpc.RetryInfo",
                            "retryDelay": "48s",
                        }
                    ]
                }
            },
            pytest.raises(TooManyRequestsError),
            id="429-retry",
        ),
        pytest.param(
            400,
            {"error": {"message": "bad request"}},
            pytest.raises(LLMError, match="400 bad request"),
            id="400-error",
        ),
    ],
)
def test_check_resp(status, json_data, ctx, make_resp):
    resp = make_resp(status, json_data)
    with ctx:
        result = _check_resp(resp)
        assert result == json_data


# @pytest.mark.parametrize(
#     "body, expected",
#     [
#         pytest.param(
#             json.dumps(
#                 {
#                     "error": {
#                         "details": [
#                             {
#                                 "@type": "type.googleapis.com/google.rpc.RetryInfo",
#                                 "retryDelay": "30.5s",
#                             }
#                         ]
#                     }
#                 }
#             ).encode(),
#             31,
#             id="extracts-delay",
#         ),
#         pytest.param(b"not json", None, id="invalid-json"),
#         pytest.param(
#             json.dumps({"error": {"details": []}}).encode(),
#             None,
#             id="no-retry-info",
#         ),
#     ],
# )
# async def test_gemini_retry_async_get_retry_after(body: bytes, expected: float | None):
#     retry = GeminiRetry()
#
#     class _Resp:
#         @property
#         async def data(self):
#             return body
#
#     assert await retry.async_get_retry_after(_Resp()) == expected


class TestGeminiChatStream:
    def test_build_body_skips_system(self):
        messages: list[Message] = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        body = GeminiChatStream.build_body(messages, 0.5)
        assert body.get("temperature") == 0.5
        assert len(body["contents"]) == 2
        assert body["contents"][0]["role"] == "user"
        assert body["contents"][1]["role"] == "model"

    def setup_method(self):
        client = MagicMock(spec=GeminiClient)
        self.stream = GeminiChatStream(client, [])

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            pytest.param(
                {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]},
                "hi",
                id="text",
            ),
            pytest.param({}, None, id="empty"),
            pytest.param(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "thinking...", "thought": True}]
                            }
                        }
                    ]
                },
                None,
                id="thought_skipped",
            ),
            pytest.param(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "thinking...", "thought": True},
                                    {"text": "answer"},
                                ]
                            }
                        }
                    ]
                },
                "answer",
                id="thought_then_text",
            ),
        ],
    )
    def test_extract_text(self, chunk: dict, expected: str | None):
        assert self.stream._extract_text(chunk) == expected

    def test_on_thought_callback(self):
        received: list[str] = []
        client = MagicMock(spec=GeminiClient)
        stream = GeminiChatStream(client, [], on_thought=received.append)
        chunk = {
            "candidates": [
                {"content": {"parts": [{"text": "pondering", "thought": True}]}}
            ]
        }
        stream._extract_text(chunk)
        assert received == ["pondering"]

    @pytest.mark.parametrize(
        "chunk, expected",
        [
            pytest.param(
                {
                    "usageMetadata": {
                        "totalTokenCount": 100,
                        "promptTokenCount": 80,
                        "candidatesTokenCount": 20,
                    }
                },
                {"total": 100, "input": 80, "output": 20},
                id="basic",
            ),
            pytest.param(
                {
                    "usageMetadata": {
                        "totalTokenCount": 100,
                        "promptTokenCount": 80,
                        "candidatesTokenCount": 20,
                        "cachedContentTokenCount": 10,
                    }
                },
                {"total": 100, "input": 80, "output": 20, "cached": 10},
                id="cached",
            ),
            pytest.param({}, None, id="no-usage"),
        ],
    )
    def test_extract_usage(self, chunk: dict, expected: dict | None):
        assert self.stream._extract_usage(chunk) == expected


class TestGeminiStream:
    """Tests for GeminiClient.stream() SSE parsing."""

    @pytest.fixture
    def client(self):
        c = MagicMock(spec=GeminiClient)
        c.provider = "gemini"
        c.model = "gemini-2.5-flash"
        c.base_url = "https://generativelanguage.googleapis.com/v1beta/"
        c.temperature = 0.2
        c._sse_url = GeminiClient._sse_url.__get__(c)
        c._build_gen_config = MagicMock(return_value={"temperature": 0.2})
        c.session = AsyncMock()
        return c

    @pytest.mark.asyncio
    async def test_yields_parsed_chunks(self, client, make_sse_event, make_sse_resp):
        chunk1 = {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}
        chunk2 = {"candidates": [{"content": {"parts": [{"text": " there"}]}}]}
        events = [
            make_sse_event(json.dumps(chunk1)),
            make_sse_event(json.dumps(chunk2)),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [c async for c in GeminiClient.stream(client, {"contents": []})]
        assert chunks == [chunk1, chunk2]

    @pytest.mark.asyncio
    async def test_skips_keepalive_frames(self, client, make_sse_event, make_sse_resp):
        chunk = {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
        events = [
            make_sse_event(""),
            make_sse_event(json.dumps(chunk)),
            make_sse_event(""),
        ]
        client.session.post.return_value = make_sse_resp(events)
        chunks = [c async for c in GeminiClient.stream(client, {"contents": []})]
        assert chunks == [chunk]

    @pytest.mark.asyncio
    async def test_no_extension_raises(self, client):
        resp = AsyncMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock(return_value=resp)
        resp.extension = None
        client.session.post.return_value = resp
        with pytest.raises(LLMError, match="SSE extension"):
            _ = [c async for c in GeminiClient.stream(client, {"contents": []})]

    @pytest.mark.asyncio
    async def test_malformed_json_raises(self, client, make_sse_resp):
        ev = MagicMock()
        ev.data = "{bad"
        ev.json = MagicMock(side_effect=ValueError("parse error"))
        client.session.post.return_value = make_sse_resp([ev])
        with pytest.raises(LLMError, match="Stream parse error"):
            _ = [c async for c in GeminiClient.stream(client, {"contents": []})]


class TestGenConfig:
    """Tests for `GeminiClient._build_gen_config` and its usage across APIs."""

    def test_without_thinking(self):
        client = GeminiClient(api_key="test")
        assert client._build_gen_config(0.7) == {"temperature": 0.7}

    def test_with_thinking(self):
        client = GeminiClient(
            api_key="test",
            thinking_config={"thinkingBudget": 1024, "includeThoughts": True},
        )
        assert client._build_gen_config(0.3) == {
            "temperature": 0.3,
            "thinkingConfig": {"thinkingBudget": 1024, "includeThoughts": True},
        }

    async def test_complete_chat_applies_thinking_config(self):
        """Regression: `complete_chat` used to hard-code generationConfig
        and dropped any `thinking_config` set on the client."""
        client = GeminiClient(
            api_key="test",
            thinking_config={"thinkingBudget": 2048, "includeThoughts": True},
        )

        recorded: dict[str, object] = {}

        async def fake_complete(body, model=None):
            recorded["body"] = body
            return (
                {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "ok"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
                {"total": 0, "input": 0, "output": 0},
            )

        client.complete = fake_complete  # type: ignore[method-assign]
        await client.complete_chat([{"role": "user", "content": "hi"}])

        body = recorded["body"]
        assert isinstance(body, dict)
        assert body["generationConfig"]["temperature"] == 0.2
        assert body["generationConfig"]["thinkingConfig"] == {
            "thinkingBudget": 2048,
            "includeThoughts": True,
        }

    async def test_complete_chat_skips_thought_parts(self):
        """Regression: `complete_chat` used to return the first text part
        even if it was flagged `thought: true`, leaking the model's
        internal reasoning into the final answer instead of the real
        response. Thought parts must be skipped (and forwarded to
        `on_thought` if set) like the streaming path already does.
        """
        thoughts: list[str] = []
        client = GeminiClient(
            api_key="test",
            thinking_config={"thinkingBudget": 2048, "includeThoughts": True},
            on_thought=thoughts.append,
        )

        async def fake_complete(body, model=None):
            return (
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": "thinking out loud...", "thought": True},
                                    {"text": "the real answer"},
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ]
                },
                {"total": 0, "input": 0, "output": 0},
            )

        client.complete = fake_complete  # type: ignore[method-assign]
        response, _ = await client.complete_chat([{"role": "user", "content": "hi"}])
        assert response["content"] == "the real answer"
        assert thoughts == ["thinking out loud..."]


# Batch


@pytest.mark.parametrize(
    "state, is_terminal, succeeded",
    [
        pytest.param("JOB_STATE_SUCCEEDED", True, True, id="succeeded"),
        pytest.param("JOB_STATE_FAILED", True, False, id="failed"),
        pytest.param("JOB_STATE_CANCELLED", True, False, id="cancelled"),
        pytest.param("JOB_STATE_RUNNING", False, False, id="running"),
        pytest.param("JOB_STATE_PENDING", False, False, id="pending"),
    ],
)
def test_batch_job_states(state: BatchState, is_terminal: bool, succeeded: bool):
    job = BatchJob(name="batches/123", state=state)
    assert job.is_terminal is is_terminal
    assert job.succeeded is succeeded


@pytest.mark.parametrize(
    "dest, expected",
    [
        pytest.param(None, None, id="no-dest"),
        pytest.param(
            {"inlined_responses": [{"key": "r1", "response": {}}]},
            [{"key": "r1", "response": {}}],
            id="with-responses",
        ),
    ],
)
def test_batch_job_inlined_responses(dest, expected):
    job = BatchJob(name="b/1", state="JOB_STATE_SUCCEEDED", dest=dest)
    assert job.inlined_responses == expected


@pytest.mark.parametrize(
    "resp, expected_content, expected_total",
    [
        pytest.param(
            {
                "key": "req-0",
                "response": {
                    "candidates": [{"content": {"parts": [{"text": "answer"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 15,
                    },
                },
            },
            "answer",
            15,
            id="full-response",
        ),
        pytest.param({"key": "r1", "response": {}}, "", 0, id="empty-response"),
    ],
)
def test_batch_result_from_inlined_response(
    resp: InlinedResponse, expected_content: str, expected_total: int
):
    result = BatchResult.from_inlined_response(resp)
    assert result.content == expected_content
    assert result.total_tokens == expected_total


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


@pytest.mark.parametrize(
    "local_type, sdk_type",
    [
        pytest.param(Part, PartDict, id="Part"),
        pytest.param(Content, ContentDict, id="Content"),
        pytest.param(ThinkingConfig, ThinkingConfigDict, id="ThinkingConfig"),
        pytest.param(GenerationConfig, GenerationConfigDict, id="GenerationConfig"),
    ],
)
def test_sdk_compat(local_type: type, sdk_type: type):
    """Verify every field in our local models maps to a field in the SDK type."""
    sdk_keys = set(get_type_hints(sdk_type))
    for key in get_type_hints(local_type):
        snake = _camel_to_snake(key)
        assert snake in sdk_keys, (
            f"{local_type.__name__}.{key} ({snake}) not in {sdk_type.__name__}"
        )
