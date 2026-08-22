from typing import TYPE_CHECKING, Any, cast

import pytest

from padwan_llm.anthropic.events import (
    error_to_anthropic,
    response_to_anthropic,
    stream_to_anthropic,
)
from padwan_llm.errors import LLMError, QuotaExceededError, TooManyRequestsError

if TYPE_CHECKING:
    from padwan_llm.openai.types import CreateChatCompletionResponse

USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 20,
    "total_tokens": 120,
    "prompt_tokens_details": {"cached_tokens": 60},
}


def _completion(message, finish_reason="stop", usage=USAGE):
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "glm-4.6",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _chunk(delta=None, finish_reason=None, usage=None):
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "glm-4.6",
        "choices": [],
    }
    if delta is not None or finish_reason is not None:
        chunk["choices"] = [
            {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        ]
    if usage is not None:
        chunk["usage"] = usage
    return chunk


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


async def _collect(chunks, model="claude-sonnet-5"):
    return [event async for event in stream_to_anthropic(_stream(chunks), model=model)]


def _to_anthropic(data: dict, *, model: str | None = None) -> dict[str, Any]:
    """response_to_anthropic untyped, for assertion access to NotRequired keys."""
    typed = cast("CreateChatCompletionResponse", data)
    return cast("dict[str, Any]", response_to_anthropic(typed, model=model))


# response_to_anthropic


def test_text_response():
    data = _completion({"role": "assistant", "content": "Hello!"})
    resp = _to_anthropic(data, model="claude-sonnet-5")
    assert resp["content"] == [{"type": "text", "text": "Hello!"}]
    assert resp["stop_reason"] == "end_turn"
    assert resp["role"] == "assistant"
    assert resp["model"] == "claude-sonnet-5"
    assert resp["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 60,
    }


def test_model_defaults_to_backend():
    data = _completion({"role": "assistant", "content": "Hello!"})
    assert _to_anthropic(data)["model"] == "glm-4.6"


@pytest.mark.parametrize(
    "arguments, expected_input",
    [
        pytest.param('{"city": "Paris"}', {"city": "Paris"}, id="json_string"),
        pytest.param({"city": "Paris"}, {"city": "Paris"}, id="dict_arguments"),
        pytest.param("", {}, id="empty"),
        pytest.param("{not json", {}, id="invalid_json"),
    ],
)
def test_tool_call_response(arguments, expected_input):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": arguments},
            }
        ],
    }
    resp = _to_anthropic(_completion(message, finish_reason="tool_calls"))
    assert resp["content"] == [
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": expected_input,
        }
    ]
    assert resp["stop_reason"] == "tool_use"


def test_reasoning_becomes_thinking_block():
    message = {
        "role": "assistant",
        "content": "Answer.",
        "reasoning_content": "step by step",
    }
    resp = _to_anthropic(_completion(message))
    assert resp["content"] == [
        {"type": "thinking", "thinking": "step by step"},
        {"type": "text", "text": "Answer."},
    ]


@pytest.mark.parametrize(
    "finish_reason, expected",
    [
        pytest.param("stop", "end_turn", id="stop"),
        pytest.param("length", "max_tokens", id="length"),
        pytest.param("content_filter", "refusal", id="content_filter"),
        pytest.param("weird_reason", "end_turn", id="unknown"),
        pytest.param(None, "end_turn", id="missing"),
    ],
)
def test_stop_reason_mapping(finish_reason, expected):
    data = _completion({"role": "assistant", "content": "x"}, finish_reason)
    assert _to_anthropic(data)["stop_reason"] == expected


# stream_to_anthropic


async def test_text_stream_event_sequence():
    chunks = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk({"content": "Hel"}),
        _chunk({"content": "lo"}),
        _chunk({}, finish_reason="stop"),
        _chunk(usage=USAGE),
    ]
    events = await _collect(chunks)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["role"] == "assistant"
    assert start["model"] == "claude-sonnet-5"
    assert start["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert events[1][1]["content_block"] == {"type": "text", "text": ""}
    text = "".join(
        e["delta"]["text"] for _, e in events if e["type"] == "content_block_delta"
    )
    assert text == "Hello"
    delta = events[-2][1]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"] == {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_input_tokens": 60,
    }


async def test_tool_call_stream():
    chunks = [
        _chunk({"role": "assistant", "content": "Checking."}),
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            }
        ),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        _chunk({}, finish_reason="tool_calls"),
    ]
    events = await _collect(chunks)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",  # text
        "content_block_delta",
        "content_block_stop",
        "content_block_start",  # tool_use
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    tool_start = events[4][1]
    assert tool_start["index"] == 1
    assert tool_start["content_block"] == {
        "type": "tool_use",
        "id": "call_abc",
        "name": "get_weather",
        "input": {},
    }
    partial = "".join(
        e["delta"]["partial_json"]
        for _, e in events
        if e["type"] == "content_block_delta"
        and e["delta"]["type"] == "input_json_delta"
    )
    assert partial == '{"city":"Paris"}'
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


async def test_parallel_tool_calls_get_separate_blocks():
    chunks = [
        _chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_a",
                        "function": {"name": "f1", "arguments": "{}"},
                    }
                ]
            }
        ),
        _chunk(
            {
                "tool_calls": [
                    {"index": 1, "function": {"name": "f2", "arguments": "{}"}}
                ]
            }
        ),
        _chunk({}, finish_reason="tool_calls"),
    ]
    events = await _collect(chunks)
    starts = [e for _, e in events if e["type"] == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1]
    assert starts[0]["content_block"]["id"] == "call_a"
    # missing id on the second call is synthesized from the OpenAI index
    assert starts[1]["content_block"]["id"] == "call_1"
    assert starts[1]["content_block"]["name"] == "f2"


async def test_reasoning_then_text_stream():
    chunks = [
        _chunk({"reasoning_content": "hmm "}),
        _chunk({"reasoning_content": "ok"}),
        _chunk({"content": "Answer"}),
        _chunk({}, finish_reason="stop"),
    ]
    events = await _collect(chunks)
    starts = [e for _, e in events if e["type"] == "content_block_start"]
    assert [s["content_block"]["type"] for s in starts] == ["thinking", "text"]
    assert [s["index"] for s in starts] == [0, 1]
    thinking = "".join(
        e["delta"]["thinking"]
        for _, e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == "thinking_delta"
    )
    assert thinking == "hmm ok"


async def test_empty_stream_still_well_formed():
    events = await _collect([])
    assert [name for name, _ in events] == [
        "message_start",
        "message_delta",
        "message_stop",
    ]
    assert events[1][1]["delta"]["stop_reason"] == "end_turn"
    assert events[1][1]["usage"] == {"input_tokens": 0, "output_tokens": 0}


# error_to_anthropic


@pytest.mark.parametrize(
    "exc, expected_status, expected_type",
    [
        pytest.param(
            TooManyRequestsError(retry_delay=30, message="slow down"),
            429,
            "rate_limit_error",
            id="rate_limit",
        ),
        pytest.param(
            QuotaExceededError(body={"error": "no credits"}),
            400,
            "invalid_request_error",
            id="quota",
        ),
        pytest.param(LLMError("openai", "backend down"), 502, "api_error", id="llm"),
        pytest.param(ValueError("boom"), 500, "api_error", id="unexpected"),
    ],
)
def test_error_to_anthropic(exc, expected_status, expected_type):
    status, body = error_to_anthropic(exc)
    assert status == expected_status
    assert body["type"] == "error"
    assert body["error"]["type"] == expected_type
    assert body["error"]["message"]
