import pytest

from padwan_llm import LLMClient

from .conftest import (
    PROMPT,
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
        text, usage = await client.complete_chat(PROMPT)

    assert "hello" in text.lower()
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
