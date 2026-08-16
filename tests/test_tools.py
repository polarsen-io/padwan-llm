from typing import cast
from unittest.mock import MagicMock

import pytest

from padwan_llm.conversation import AssistantToolMessage, Message, ToolResultMessage
from padwan_llm.gemini.client import GeminiChatStream, GeminiClient
from padwan_llm.gemini.models import FunctionResponsePart
from padwan_llm.gemini.tools import GeminiToolMixin
from padwan_llm.models import (
    ChatResponse,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
)
from padwan_llm.openai.client import OpenAIChatStream, OpenAIClient
from padwan_llm.openai.tools import OpenAIToolMixin

SAMPLE_TOOL: ToolDefinition = {
    "name": "get_weather",
    "description": "Get the weather for a city",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

SAMPLE_TOOL_CALL = ToolCall(
    id="call_1",
    type="function",
    function=ToolCallFunction(name="get_weather", arguments='{"city": "Paris"}'),
)


# ChatResponse construction


@pytest.mark.parametrize(
    "content, finish_reason, tool_calls, expected_keys",
    [
        pytest.param(
            "Hello!",
            "stop",
            None,
            {"content", "finish_reason"},
            id="text-only",
        ),
        pytest.param(
            None,
            "tool_calls",
            [SAMPLE_TOOL_CALL],
            {"content", "finish_reason", "tool_calls"},
            id="with-tool-calls",
        ),
    ],
)
def test_chat_response(content, finish_reason, tool_calls, expected_keys):
    resp: ChatResponse = {"content": content, "finish_reason": finish_reason}
    if tool_calls:
        resp["tool_calls"] = tool_calls
    assert set(resp.keys()) == expected_keys
    assert resp["finish_reason"] == finish_reason


# OpenAI tool call extraction


@pytest.mark.parametrize(
    "raw_calls, expected",
    [
        pytest.param(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
            [SAMPLE_TOOL_CALL],
            id="string-arguments",
        ),
        pytest.param(
            [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": {"city": "Paris"},
                    },
                }
            ],
            [
                ToolCall(
                    id="call_2",
                    type="function",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    ),
                )
            ],
            id="mistral-dict-arguments",
        ),
        pytest.param(
            [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "func_a", "arguments": "{}"},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "func_b", "arguments": '{"x": 1}'},
                },
            ],
            [
                ToolCall(
                    id="call_a",
                    type="function",
                    function=ToolCallFunction(name="func_a", arguments="{}"),
                ),
                ToolCall(
                    id="call_b",
                    type="function",
                    function=ToolCallFunction(name="func_b", arguments='{"x": 1}'),
                ),
            ],
            id="multiple-calls",
        ),
    ],
)
def test_extract_tool_calls(raw_calls, expected):
    assert OpenAIToolMixin._extract_tool_calls(raw_calls) == expected


# OpenAI streaming tool call delta reassembly


class TestOpenAIStreamToolCallDeltas:
    def setup_method(self):
        client = MagicMock(spec=OpenAIClient)
        self.stream = OpenAIChatStream(client, [])

    @pytest.mark.parametrize(
        "chunks, expected_pending",
        [
            pytest.param(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": '{"city":',
                                            },
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
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {
                                                "arguments": ' "Paris"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                ],
                {
                    0: {
                        "id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    }
                },
                id="single-tool-two-chunks",
            ),
            pytest.param(
                [
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_a",
                                            "function": {
                                                "name": "func_a",
                                                "arguments": "{}",
                                            },
                                        },
                                        {
                                            "index": 1,
                                            "id": "call_b",
                                            "function": {
                                                "name": "func_b",
                                                "arguments": '{"x": 1}',
                                            },
                                        },
                                    ]
                                }
                            }
                        ]
                    },
                ],
                {
                    0: {"id": "call_a", "name": "func_a", "arguments": "{}"},
                    1: {"id": "call_b", "name": "func_b", "arguments": '{"x": 1}'},
                },
                id="parallel-tools-single-chunk",
            ),
            pytest.param(
                [{"choices": [{"delta": {"content": "hello"}}]}],
                {},
                id="text-only-no-tool-calls",
            ),
            pytest.param(
                [{}],
                {},
                id="empty-chunk",
            ),
        ],
    )
    def test_accumulate(self, chunks, expected_pending):
        pending: dict[int, dict[str, str]] = {}
        for chunk in chunks:
            OpenAIChatStream._accumulate_tool_call_deltas(chunk, pending)
        assert pending == expected_pending

    def test_pending_to_tool_calls(self):
        """Verify pending dict is correctly assembled into sorted ToolCall list."""
        pending = {
            1: {"id": "call_b", "name": "func_b", "arguments": '{"y": 2}'},
            0: {"id": "call_a", "name": "func_a", "arguments": "{}"},
        }
        result = [
            ToolCall(
                id=tc["id"],
                type="function",
                function=ToolCallFunction(name=tc["name"], arguments=tc["arguments"]),
            )
            for tc in (pending[i] for i in sorted(pending))
        ]
        assert result[0]["id"] == "call_a"
        assert result[1]["id"] == "call_b"
        assert len(result) == 2


# Gemini function call extraction


@pytest.mark.parametrize(
    "parts, expected",
    [
        pytest.param(
            [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
            [
                ToolCall(
                    id="call_0",
                    type="function",
                    function=ToolCallFunction(
                        name="get_weather",
                        arguments='{"city": "Paris"}',
                    ),
                )
            ],
            id="single-function-call",
        ),
        pytest.param(
            [{"text": "Hello"}],
            [],
            id="text-only-no-calls",
        ),
        pytest.param(
            [
                {"functionCall": {"name": "func_a", "args": {}}},
                {"text": "Thinking..."},
                {"functionCall": {"name": "func_b", "args": {"x": 1}}},
            ],
            [
                ToolCall(
                    id="call_0",
                    type="function",
                    function=ToolCallFunction(name="func_a", arguments="{}"),
                ),
                ToolCall(
                    id="call_2",
                    type="function",
                    function=ToolCallFunction(name="func_b", arguments='{"x": 1}'),
                ),
            ],
            id="mixed-parts-skips-text",
        ),
        pytest.param(
            [{"functionCall": {"name": "no_args"}}],
            [
                ToolCall(
                    id="call_0",
                    type="function",
                    function=ToolCallFunction(name="no_args", arguments="{}"),
                )
            ],
            id="missing-args-defaults-empty",
        ),
    ],
)
def test_extract_gemini_tool_calls(parts, expected):
    assert GeminiToolMixin._extract_gemini_tool_calls(parts) == expected


# Gemini stream tool call extraction


class TestGeminiStreamToolCalls:
    def setup_method(self):
        client = MagicMock(spec=GeminiClient)
        self.stream = GeminiChatStream(client, [])

    @pytest.mark.parametrize(
        "chunk, expected_count",
        [
            pytest.param(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {
                                            "name": "get_weather",
                                            "args": {"city": "Paris"},
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
                1,
                id="with-function-call",
            ),
            pytest.param(
                {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]},
                0,
                id="text-only",
            ),
            pytest.param({}, 0, id="empty"),
            pytest.param({"candidates": []}, 0, id="no-candidates"),
        ],
    )
    def test_extract_tool_calls_from_chunk(self, chunk, expected_count):
        result = GeminiChatStream._extract_tool_calls_from_chunk(chunk)
        assert len(result) == expected_count


# ToolDefinition to provider format


class TestToolDefinitionMapping:
    def test_to_openai(self):
        result = OpenAIToolMixin._tools_to_openai([SAMPLE_TOOL])
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        fn = tool["function"]
        assert fn["name"] == "get_weather"
        assert fn.get("description") == "Get the weather for a city"
        assert fn.get("parameters") == SAMPLE_TOOL["parameters"]

    def test_to_gemini(self):
        result = GeminiToolMixin._tools_to_gemini([SAMPLE_TOOL])
        assert len(result) == 1
        decls = result[0]["function_declarations"]
        assert len(decls) == 1
        decl = decls[0]
        assert decl["name"] == "get_weather"
        assert decl["description"] == "Get the weather for a city"
        assert decl.get("parameters") == SAMPLE_TOOL["parameters"]

    def test_multiple_tools(self):
        tool2: ToolDefinition = {
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
        openai_result = OpenAIToolMixin._tools_to_openai([SAMPLE_TOOL, tool2])
        assert len(openai_result) == 2

        gemini_result = GeminiToolMixin._tools_to_gemini([SAMPLE_TOOL, tool2])
        assert len(gemini_result) == 1
        assert len(gemini_result[0]["function_declarations"]) == 2


# Gemini build_body with tool message types


class TestGeminiBuildBodyToolMessages:
    def test_tool_result_message(self):
        messages: list = [
            Message(role="user", content="What's the weather?"),
            AssistantToolMessage(
                role="assistant",
                content=None,
                tool_calls=[SAMPLE_TOOL_CALL],
            ),
            ToolResultMessage(
                role="tool",
                tool_call_id="call_1",
                name="get_weather",
                content='{"temperature": 20}',
            ),
        ]
        body = GeminiChatStream.build_body(messages, 0.5)
        assert len(body["contents"]) == 3

        # User message
        assert body["contents"][0]["role"] == "user"

        # Assistant tool call → model with functionCall part
        model_msg = body["contents"][1]
        assert model_msg["role"] == "model"
        assert "functionCall" in model_msg["parts"][0]
        assert model_msg["parts"][0]["functionCall"]["name"] == "get_weather"

        # Tool result → user with functionResponse part
        tool_resp = body["contents"][2]
        assert tool_resp["role"] == "user"
        assert "functionResponse" in tool_resp["parts"][0]
        assert tool_resp["parts"][0]["functionResponse"]["name"] == "get_weather"
        assert tool_resp["parts"][0]["functionResponse"]["response"] == {
            "temperature": 20
        }

    def test_assistant_tool_message_with_text(self):
        messages: list = [
            AssistantToolMessage(
                role="assistant",
                content="Let me check the weather.",
                tool_calls=[SAMPLE_TOOL_CALL],
            ),
        ]
        body = GeminiChatStream.build_body(messages, 0.5)
        parts = body["contents"][0]["parts"]
        # Text part first, then functionCall
        assert parts[0] == {"text": "Let me check the weather."}
        assert "functionCall" in parts[1]

    def test_tool_result_non_json_content(self):
        messages: list = [
            ToolResultMessage(
                role="tool",
                tool_call_id="call_1",
                name="get_weather",
                content="plain text result",
            ),
        ]
        body = GeminiChatStream.build_body(messages, 0.5)
        part = cast(FunctionResponsePart, body["contents"][0]["parts"][0])
        assert part["functionResponse"]["response"] == {"result": "plain text result"}

    def test_tools_parameter(self):
        messages: list = [Message(role="user", content="Hi")]
        body = GeminiChatStream.build_body(messages, 0.5, tools=[SAMPLE_TOOL])
        assert "tools" in body
        assert body["tools"][0]["function_declarations"][0]["name"] == "get_weather"

    def test_no_tools_parameter(self):
        messages: list = [Message(role="user", content="Hi")]
        body = GeminiChatStream.build_body(messages, 0.5)
        assert "tools" not in body
