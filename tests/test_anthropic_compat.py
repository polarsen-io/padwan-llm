import json
from typing import Any, cast

import pytest

from padwan_llm.anthropic.compat import messages_to_openai
from padwan_llm.anthropic.models import MessagesBody

BASE64_PNG = "iVBORw0KGgo="


def _body(**overrides) -> MessagesBody:
    body: MessagesBody = {
        "model": "claude-sonnet-5",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(cast(Any, overrides))
    return body


def _to_openai(body: MessagesBody, *, model: str | None = None) -> dict[str, Any]:
    """messages_to_openai untyped, for assertion access to NotRequired keys."""
    return cast("dict[str, Any]", messages_to_openai(body, model=model))


@pytest.mark.parametrize(
    "model, expected_key",
    [
        pytest.param("glm-4.6", "max_tokens", id="openai_compatible_backend"),
        pytest.param("gpt-5-mini", "max_completion_tokens", id="openai_reasoning"),
    ],
)
def test_max_tokens_key_per_backend(model, expected_key):
    request = _to_openai(_body(), model=model)
    assert request["model"] == model
    assert request[expected_key] == 1024


def test_model_kept_when_not_overridden():
    request = _to_openai(_body())
    assert request["model"] == "claude-sonnet-5"


@pytest.mark.parametrize(
    "system, expected",
    [
        pytest.param("You are helpful.", "You are helpful.", id="string"),
        pytest.param(
            [
                {"type": "text", "text": "You are Claude Code."},
                {
                    "type": "text",
                    "text": "Extra context.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "You are Claude Code.\n\nExtra context.",
            id="blocks_with_cache_control",
        ),
    ],
)
def test_system_flattened_to_system_message(system, expected):
    request = _to_openai(_body(system=system), model="glm-4.6")
    assert request["messages"][0] == {"role": "system", "content": expected}
    assert request["messages"][1] == {"role": "user", "content": "hello"}


def test_empty_system_omitted():
    request = _to_openai(_body(system=""), model="glm-4.6")
    assert request["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    "blocks, expected",
    [
        pytest.param(
            [{"type": "text", "text": "describe this"}],
            [{"role": "user", "content": "describe this"}],
            id="single_text_block_flattened_to_string",
        ),
        pytest.param(
            [
                {"type": "text", "text": "describe this"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": BASE64_PNG,
                    },
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{BASE64_PNG}"},
                        },
                    ],
                }
            ],
            id="text_and_base64_image",
        ),
        pytest.param(
            [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/a.png"},
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        }
                    ],
                }
            ],
            id="url_image",
        ),
    ],
)
def test_user_content_blocks(blocks, expected):
    request = _to_openai(
        _body(messages=[{"role": "user", "content": blocks}]), model="glm-4.6"
    )
    assert request["messages"] == expected


@pytest.mark.parametrize(
    "content, expected_text",
    [
        pytest.param("42 files", "42 files", id="string_content"),
        pytest.param(
            [
                {"type": "text", "text": "line 1"},
                {"type": "text", "text": "line 2"},
            ],
            "line 1\nline 2",
            id="text_blocks_joined",
        ),
        pytest.param(None, "", id="missing_content"),
    ],
)
def test_tool_result_becomes_tool_message(content, expected_text):
    block = {"type": "tool_result", "tool_use_id": "toolu_1", "content": content}
    request = _to_openai(
        _body(messages=[{"role": "user", "content": [block]}]), model="glm-4.6"
    )
    assert request["messages"] == [
        {"role": "tool", "tool_call_id": "toolu_1", "content": expected_text}
    ]


def test_tool_result_image_forwarded_as_user_message():
    blocks = [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [
                {"type": "text", "text": "screenshot taken"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": BASE64_PNG,
                    },
                },
            ],
        },
        {"type": "text", "text": "what do you see?"},
    ]
    request = _to_openai(
        _body(messages=[{"role": "user", "content": blocks}]), model="glm-4.6"
    )
    tool_msg, user_msg = request["messages"]
    assert tool_msg == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "screenshot taken",
    }
    assert user_msg["role"] == "user"
    assert user_msg["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{BASE64_PNG}"},
        },
        {"type": "text", "text": "what do you see?"},
    ]


def test_assistant_blocks_with_tool_use_and_thinking():
    messages = [
        {"role": "user", "content": "weather in Paris?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me check", "signature": "sig"},
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18°C"}
            ],
        },
    ]
    request = _to_openai(_body(messages=messages), model="glm-4.6")
    assistant = request["messages"][1]
    assert assistant["content"] == "Checking."
    (tool_call,) = assistant["tool_calls"]
    assert tool_call["id"] == "toolu_1"
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "Paris"}
    assert request["messages"][2]["role"] == "tool"


def test_tools_converted_and_server_tools_skipped():
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather.",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": {"type": "ephemeral"},
        },
        {"name": "web_search", "type": "web_search_20250305"},
    ]
    request = _to_openai(_body(tools=tools), model="glm-4.6")
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
                "description": "Get weather.",
            },
        }
    ]


def test_all_server_tools_omits_tools_and_tool_choice():
    tools = [{"name": "web_search", "type": "web_search_20250305"}]
    request = _to_openai(
        _body(tools=tools, tool_choice={"type": "auto"}), model="glm-4.6"
    )
    assert "tools" not in request
    assert "tool_choice" not in request


WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get weather.",
    "input_schema": {"type": "object"},
}


@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        pytest.param({"type": "auto"}, "auto", id="auto"),
        pytest.param({"type": "any"}, "required", id="any"),
        pytest.param({"type": "none"}, "none", id="none"),
        pytest.param(
            {"type": "tool", "name": "get_weather"},
            {"type": "function", "function": {"name": "get_weather"}},
            id="named_tool",
        ),
    ],
)
def test_tool_choice_mapping(tool_choice, expected):
    request = _to_openai(
        _body(tools=[WEATHER_TOOL], tool_choice=tool_choice), model="glm-4.6"
    )
    assert request["tool_choice"] == expected


def test_disable_parallel_tool_use():
    request = _to_openai(
        _body(
            tools=[WEATHER_TOOL],
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        ),
        model="glm-4.6",
    )
    assert request["parallel_tool_calls"] is False


def test_sampling_params_and_dropped_fields():
    request = _to_openai(
        _body(
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            stop_sequences=["END"],
            metadata={"user_id": "u1"},
            thinking={"type": "enabled", "budget_tokens": 1000},
        ),
        model="glm-4.6",
    )
    assert request["temperature"] == 0.5
    assert request["top_p"] == 0.9
    assert request["stop"] == ["END"]
    for dropped in ("top_k", "metadata", "thinking"):
        assert dropped not in request
