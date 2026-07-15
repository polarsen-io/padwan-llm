import json

import pytest

from padwan_llm import AnthropicClient

from .conftest import PROMPT, TOOL_PROMPT, WEATHER_TOOL, skip_no_anthropic

pytestmark = [pytest.mark.e2e, skip_no_anthropic]

MODEL = "claude-haiku-4-5"


async def test_complete_chat() -> None:
    async with AnthropicClient(model=MODEL) as client:
        response, usage = await client.complete_chat(PROMPT)

    assert response["content"]
    assert "hello" in response["content"].lower()
    assert response["finish_reason"] == "stop"
    assert usage["input"] > 0
    assert usage["output"] > 0


async def test_stream_chat() -> None:
    async with AnthropicClient(model=MODEL) as client:
        stream = client.stream_chat(PROMPT)
        text = "".join([chunk async for chunk in stream])

    assert "hello" in text.lower()
    assert stream.usage is not None
    assert stream.usage["output"] > 0


async def test_tool_call_roundtrip() -> None:
    async with AnthropicClient(model=MODEL) as client:
        response, _ = await client.complete_chat(TOOL_PROMPT, tools=[WEATHER_TOOL])
        assert response["finish_reason"] == "tool_calls"
        tool_calls = response.get("tool_calls")
        assert tool_calls
        call = tool_calls[0]
        assert call["function"]["name"] == "get_weather"
        assert "paris" in json.loads(call["function"]["arguments"])["city"].lower()

        # Send the result back and get a final text answer
        response2, _ = await client.complete_chat(
            [
                *TOOL_PROMPT,
                {
                    "role": "assistant",
                    "content": response["content"],
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": "get_weather",
                    "content": "18°C and sunny",
                },
            ],
            tools=[WEATHER_TOOL],
        )

    assert response2["content"]
    assert "18" in response2["content"]


async def test_stream_chat_tool_calls() -> None:
    async with AnthropicClient(model=MODEL) as client:
        stream = client.stream_chat(TOOL_PROMPT, tools=[WEATHER_TOOL])
        _ = [chunk async for chunk in stream]

    assert stream.tool_calls
    assert stream.tool_calls[0]["function"]["name"] == "get_weather"
