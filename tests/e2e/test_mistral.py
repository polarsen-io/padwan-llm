import pytest

from padwan_llm import MistralClient

from .conftest import AUDIO_FIXTURE, skip_no_mistral

pytestmark = [pytest.mark.e2e, skip_no_mistral]


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
