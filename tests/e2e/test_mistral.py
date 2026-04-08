import pytest

from padwan_llm import MistralClient

from .conftest import AUDIO_FIXTURE, skip_no_mistral

pytestmark = [pytest.mark.e2e, skip_no_mistral]


async def test_stream_thought_callback() -> None:
    """Mistral's Magistral models surface reasoning as `ThinkChunk`
    items inside a structured `delta.content` array. The lifted
    `on_thought` callback must receive that text and the regular
    stream must only contain the final answer.
    """
    received: list[str] = []
    async with MistralClient(
        model="magistral-small-latest",
        on_thought=received.append,
    ) as client:
        stream = client.stream_chat([{"role": "user", "content": "What is 7 * 8?"}])
        text = "".join([chunk async for chunk in stream])

    assert text, "no text returned"
    assert "56" in text
    assert received, "no thought chunks received from Magistral"


async def test_complete_chat_thought_callback() -> None:
    """Same contract for the non-streaming path."""
    received: list[str] = []
    async with MistralClient(
        model="magistral-small-latest",
        on_thought=received.append,
    ) as client:
        response, _ = await client.complete_chat(
            [{"role": "user", "content": "What is 7 * 8?"}]
        )

    assert response["content"], "no text returned"
    assert "56" in response["content"]
    assert received, "no thought chunks forwarded to on_thought"


async def test_embeddings() -> None:
    async with MistralClient() as client:
        resp = await client.fetch_embeddings("Hello, world!")
        assert resp["data"]
        assert len(resp["data"][0].get("embedding", [])) > 0
        assert resp["usage"]["total_tokens"] > 0


async def test_embeddings_batch() -> None:
    async with MistralClient() as client:
        resp = await client.fetch_embeddings(["Hello", "World"])
        assert len(resp["data"]) == 2


@pytest.mark.skipif(not AUDIO_FIXTURE.exists(), reason="audio fixture not found")
async def test_transcribe() -> None:
    async with MistralClient() as client:
        resp = await client.transcribe(file=AUDIO_FIXTURE)
        assert "text" in resp
        assert resp["text"]
        audio_seconds = resp["usage"].get("prompt_audio_seconds")
        assert audio_seconds is not None and audio_seconds > 0
