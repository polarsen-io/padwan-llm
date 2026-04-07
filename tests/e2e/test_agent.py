from typing import Any

import pytest

from padwan_llm import AgentSession, LLMClient, McpTool

from .conftest import (
    skip_no_gemini,
    skip_no_grok,
    skip_no_mistral,
    skip_no_openai,
)

pytestmark = pytest.mark.e2e


def _make_weather_tool(call_log: list[dict[str, Any]]) -> McpTool:
    async def get_weather(args: dict[str, Any]) -> str:
        call_log.append(args)
        city = args.get("city", "unknown")
        return f"It is sunny and 23°C in {city}."

    return McpTool(
        name="get_weather",
        description="Get the current weather for a given city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name"}},
            "required": ["city"],
        },
        handler=get_weather,
    )


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_agent_session_tool_round_trip(model: str) -> None:
    """End-to-end: AgentSession dispatches a local tool against a real LLM."""
    call_log: list[dict[str, Any]] = []
    weather = _make_weather_tool(call_log)

    async with AgentSession(
        client=LLMClient(model=model),
        mcp_tools=[weather],
        system="Use the provided tools to answer questions about the weather.",
    ) as session:
        text = await session.send("What is the weather in Paris right now?")

    assert call_log, "the model never invoked the weather tool"
    assert any(args.get("city", "").lower() == "paris" for args in call_log)
    assert "paris" in text.lower()
    assert "sunny" in text.lower() or "23" in text
    assert session.total_usage["total"] > 0


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("gemini-2.5-flash", id="gemini", marks=skip_no_gemini),
        pytest.param("gpt-4o-mini", id="openai", marks=skip_no_openai),
        pytest.param("mistral-small-latest", id="mistral", marks=skip_no_mistral),
        pytest.param("grok-4-1-fast-non-reasoning", id="grok", marks=skip_no_grok),
    ],
)
async def test_agent_session_streams_text(model: str) -> None:
    """End-to-end: AgentSession.stream yields chunks for a plain text answer."""
    chunks: list[str] = []
    async with AgentSession(client=LLMClient(model=model)) as session:
        async for chunk in session.stream("Reply with only the word 'hello'."):
            chunks.append(chunk)

    text = "".join(chunks)
    assert "hello" in text.lower()
    assert session.total_usage["total"] > 0
