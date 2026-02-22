import json

import pytest

from padwan_llm import LLMClient

from .conftest import (
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
        pytest.param("grok-3-mini-fast", id="grok", marks=skip_no_grok),
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
        pytest.param("grok-3-mini-fast", id="grok", marks=skip_no_grok),
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


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-3-mini-fast", id="grok", marks=skip_no_grok),
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
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-3-mini-fast", id="grok", marks=skip_no_grok),
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
