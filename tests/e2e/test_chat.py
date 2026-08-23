import json

import pytest

from padwan_llm import LLMClient, content_parts
from padwan_llm.conversation import AssistantToolMessage, Message, ToolResultMessage

from .conftest import (
    AUDIO_FIXTURE,
    PROMPT,
    TOOL_PROMPT,
    WEATHER_TOOL,
    skip_no_gemini,
    skip_no_grok,
    skip_no_mistral,
    skip_no_openai,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_complete_chat(model: str) -> None:
    client = LLMClient(model=model)
    async with client:
        response, usage = await client.complete_chat(PROMPT)

    assert response["content"] is not None
    assert "hello" in response["content"].lower()
    assert response["finish_reason"] == "stop"
    assert usage["total"] > 0
    assert usage["input"] > 0
    assert usage["output"] > 0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_stream_chat(model: str) -> None:
    chunks: list[str] = []
    async with LLMClient(model=model) as client:
        stream = client.stream_chat(PROMPT)
        async for chunk in stream:
            chunks.append(chunk)

    text = "".join(chunks)
    assert "hello" in text.lower()
    assert len(chunks) >= 1
    assert stream.usage is not None
    assert stream.usage["total"] > 0


@pytest.mark.skipif(not AUDIO_FIXTURE.exists(), reason="audio fixture not found")
@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param("gpt-audio", id="openai", marks=skip_no_openai),
        pytest.param("voxtral-small-latest", id="mistral", marks=skip_no_mistral),
    ],
)
async def test_complete_chat_audio(model: str) -> None:
    messages = [
        Message(
            role="user",
            content=content_parts(
                "Describe what you hear in one short sentence.", AUDIO_FIXTURE
            ),
        )
    ]
    async with LLMClient(model=model) as client:
        response, usage = await client.complete_chat(messages)

    assert response["content"]
    assert usage["total"] > 0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param(
            "gemini-3-flash-preview", id="gemini-thinking", marks=skip_no_gemini
        ),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_complete_chat_tool_call(model: str) -> None:
    async with LLMClient(model=model) as client:
        response, usage = await client.complete_chat(TOOL_PROMPT, tools=[WEATHER_TOOL])

    assert response["finish_reason"] == "tool_calls"
    tool_calls = response.get("tool_calls")
    assert tool_calls
    tc = tool_calls[0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    args = json.loads(tc["function"]["arguments"])
    assert "city" in args
    assert usage["total"] > 0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param(
            "gemini-3-flash-preview", id="gemini-thinking", marks=skip_no_gemini
        ),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_stream_chat_tool_call(model: str) -> None:
    async with LLMClient(model=model) as client:
        stream = client.stream_chat(TOOL_PROMPT, tools=[WEATHER_TOOL])
        async for _ in stream:
            pass

    assert stream.tool_calls
    tc = stream.tool_calls[0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_weather"
    args = json.loads(tc["function"]["arguments"])
    assert "city" in args
    assert stream.usage is not None
    assert stream.usage["total"] > 0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param(
            "gemini-3-flash-preview", id="gemini-thinking", marks=skip_no_gemini
        ),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_stream_chat_tool_round_trip(model: str) -> None:
    """Full tool call round-trip: model calls tool, we reply, model gives final answer."""
    async with LLMClient(model=model) as client:
        # Round 1: get tool call
        stream = client.stream_chat(TOOL_PROMPT, tools=[WEATHER_TOOL])
        async for _ in stream:
            pass

        assert stream.tool_calls, "Expected a tool call in round 1"
        tc = stream.tool_calls[0]

        # Round 2: send tool result, expect final text response
        messages = [
            *TOOL_PROMPT,
            AssistantToolMessage(
                role="assistant", content=None, tool_calls=stream.tool_calls
            ),
            ToolResultMessage(
                role="tool",
                tool_call_id=tc["id"],
                name=tc["function"]["name"],
                content='{"temperature": "18°C", "condition": "sunny"}',
            ),
        ]
        stream2 = client.stream_chat(messages)
        chunks: list[str] = []
        async for chunk in stream2:
            chunks.append(chunk)

        text = "".join(chunks)
        assert text, "Expected a final text response after tool result"
