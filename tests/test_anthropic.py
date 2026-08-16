import json
from contextlib import nullcontext
from typing import get_type_hints
from unittest.mock import AsyncMock

import pytest
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    Usage,
)
from anthropic.types import (
    Message as SdkMessage,
)
from anthropic.types.message_create_params import MessageCreateParamsStreaming

from padwan_llm.anthropic.client import (
    AnthropicClient,
    _check_resp,
    _usage_from_anthropic,
    is_anthropic_model,
)
from padwan_llm.anthropic.models import (
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicTool,
    AnthropicUsage,
    MessagesBody,
    MessagesResponse,
)
from padwan_llm.client import LLMClient
from padwan_llm.errors import LLMError, TooManyRequestsError
from padwan_llm.models import ToolDefinition

WEATHER_TOOL: ToolDefinition = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


@pytest.mark.parametrize(
    "model, expected",
    [
        pytest.param("claude-sonnet-5", True, id="sonnet"),
        pytest.param("claude-opus-4-8", True, id="opus"),
        pytest.param("gpt-4o", False, id="openai"),
        pytest.param(None, False, id="none"),
    ],
)
def test_is_anthropic_model(model, expected):
    assert is_anthropic_model(model) is expected


def test_factory_dispatch(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    client = LLMClient("claude-sonnet-5")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-sonnet-5"


@pytest.mark.parametrize(
    "status, json_data, headers, ctx",
    [
        pytest.param(200, {"content": []}, None, nullcontext(), id="success"),
        pytest.param(
            429,
            {
                "type": "error",
                "error": {"type": "rate_limit_error", "message": "slow down"},
            },
            {"retry-after": "12"},
            pytest.raises(TooManyRequestsError),
            id="429-rate-limit",
        ),
        pytest.param(
            400,
            {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad request"},
            },
            None,
            pytest.raises(LLMError, match="400 bad request"),
            id="400-error",
        ),
    ],
)
def test_check_resp(status, json_data, headers, ctx, make_resp):
    resp = make_resp(status, json_data, headers)
    with ctx:
        assert _check_resp(resp) == json_data


@pytest.mark.parametrize(
    "usage, expected",
    [
        pytest.param(
            {"input_tokens": 10, "output_tokens": 5},
            {"total": 15, "input": 10, "output": 5},
            id="plain",
        ),
        pytest.param(
            {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 8},
            {"total": 15, "input": 10, "output": 5, "cached": 8},
            id="cached",
        ),
        pytest.param(None, {"total": 0, "input": 0, "output": 0}, id="missing"),
    ],
)
def test_usage_from_anthropic(usage, expected):
    assert _usage_from_anthropic(usage) == expected


class TestBuildBody:
    def test_system_and_tools(self):
        client = AnthropicClient(
            api_key="test", model="claude-sonnet-5", max_tokens=512
        )
        body = client.build_body(
            [
                {"role": "system", "content": "You are Obi."},
                {"role": "user", "content": "hi"},
            ],
            tools=[WEATHER_TOOL],
        )
        assert body == {
            "model": "claude-sonnet-5",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "hi"}],
            "system": "You are Obi.",
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "input_schema": WEATHER_TOOL["parameters"],
                }
            ],
        }

    def test_tool_flow_messages(self):
        client = AnthropicClient(api_key="test")
        body = client.build_body(
            [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_1",
                    "name": "get_weather",
                    "content": "18°C",
                },
            ]
        )
        assert body["messages"] == [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18°C"}
                ],
            },
        ]

    def test_no_model_raises(self):
        client = AnthropicClient(api_key="test", model=None)
        with pytest.raises(LLMError, match="No model specified"):
            client.build_body([{"role": "user", "content": "hi"}])


class TestCompleteChat:
    @pytest.mark.parametrize(
        "data, expected, ctx",
        [
            pytest.param(
                {
                    "content": [{"type": "text", "text": "Hello!"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                {"content": "Hello!", "finish_reason": "stop"},
                nullcontext(),
                id="text",
            ),
            pytest.param(
                {
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                {
                    "content": "Let me check.",
                    "finish_reason": "tool_calls",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                nullcontext(),
                id="tool-use",
            ),
            pytest.param(
                {"content": [], "stop_reason": "refusal", "usage": {}},
                None,
                pytest.raises(LLMError, match="refused"),
                id="refusal",
            ),
        ],
    )
    async def test_complete_chat(self, data, expected, ctx, make_resp):
        client = AnthropicClient(api_key="test")
        session = AsyncMock()
        client._session = session
        session.post.return_value = make_resp(200, data)
        with ctx:
            response, usage = await client.complete_chat(
                [{"role": "user", "content": "hi"}]
            )
            assert response == expected
            assert usage["input"] == data["usage"].get("input_tokens", 0)

    async def test_thinking_forwarded_and_skipped(self, make_resp):
        thoughts: list[str] = []
        client = AnthropicClient(api_key="test", on_thought=thoughts.append)
        session = AsyncMock()
        client._session = session
        session.post.return_value = make_resp(
            200,
            {
                "content": [
                    {"type": "thinking", "thinking": "reasoning...", "signature": "s"},
                    {"type": "text", "text": "42"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )
        response, _ = await client.complete_chat([{"role": "user", "content": "6*7?"}])
        assert response["content"] == "42"
        assert thoughts == ["reasoning..."]


class TestStreamChat:
    def _events(self, make_sse_event, payloads):
        return [make_sse_event(json.dumps(p)) for p in payloads]

    async def test_text_tools_and_usage(self, make_sse_event, make_sse_resp):
        client = AnthropicClient(api_key="test")
        session = AsyncMock()
        client._session = session
        session.post.return_value = make_sse_resp(
            self._events(
                make_sse_event,
                [
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 7}},
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "Hi"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": " there"},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {
                        "type": "content_block_start",
                        "index": 1,
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"city": ',
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '"Paris"}',
                        },
                    },
                    {"type": "content_block_stop", "index": 1},
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 9},
                    },
                    {"type": "message_stop"},
                ],
            )
        )
        stream = client.stream_chat(
            [{"role": "user", "content": "hi"}], tools=[WEATHER_TOOL]
        )
        chunks = [c async for c in stream]
        assert chunks == ["Hi", " there"]
        assert stream.usage == {"total": 16, "input": 7, "output": 9}
        assert stream.tool_calls == [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ]

    async def test_thinking_delta_forwarded(self, make_sse_event, make_sse_resp):
        thoughts: list[str] = []
        client = AnthropicClient(api_key="test", on_thought=thoughts.append)
        session = AsyncMock()
        client._session = session
        session.post.return_value = make_sse_resp(
            self._events(
                make_sse_event,
                [
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "thinking_delta", "thinking": "hmm"},
                    },
                    {
                        "type": "content_block_delta",
                        "index": 1,
                        "delta": {"type": "text_delta", "text": "ok"},
                    },
                ],
            )
        )
        stream = client.stream_chat([{"role": "user", "content": "hi"}])
        chunks = [c async for c in stream]
        assert chunks == ["ok"]
        assert thoughts == ["hmm"]

    async def test_error_event_raises(self, make_sse_event, make_sse_resp):
        client = AnthropicClient(api_key="test")
        session = AsyncMock()
        client._session = session
        session.post.return_value = make_sse_resp(
            self._events(
                make_sse_event,
                [
                    {
                        "type": "error",
                        "error": {"type": "overloaded_error", "message": "overloaded"},
                    }
                ],
            )
        )
        stream = client.stream_chat([{"role": "user", "content": "hi"}])
        with pytest.raises(LLMError, match="overloaded"):
            _ = [c async for c in stream]


_SDK_BLOCK_TYPES = (
    TextBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    ImageBlockParam,
    ThinkingBlockParam,
)


@pytest.mark.parametrize(
    "local_type, sdk_keys",
    [
        pytest.param(AnthropicTool, set(get_type_hints(ToolParam)), id="Tool"),
        pytest.param(AnthropicMessage, set(get_type_hints(MessageParam)), id="Message"),
        pytest.param(AnthropicUsage, set(Usage.model_fields), id="Usage"),
        pytest.param(
            MessagesResponse, set(SdkMessage.model_fields), id="MessagesResponse"
        ),
        pytest.param(
            MessagesBody,
            set(get_type_hints(MessageCreateParamsStreaming)),
            id="MessagesBody",
        ),
        pytest.param(
            AnthropicContentBlock,
            set().union(*(get_type_hints(t) for t in _SDK_BLOCK_TYPES)),
            id="ContentBlock",
        ),
    ],
)
def test_sdk_compat(local_type: type, sdk_keys: set[str]):
    """Verify every field in our local models maps to a field in the anthropic SDK type."""
    for key in get_type_hints(local_type):
        assert key in sdk_keys, f"{local_type.__name__}.{key} not in SDK type"
