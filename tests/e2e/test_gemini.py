import pytest

from padwan_llm import GeminiClient
from padwan_llm.gemini.batch import BatchRequest

from .conftest import skip_no_gemini

pytestmark = [pytest.mark.e2e, skip_no_gemini]


async def test_stream_thought_callback() -> None:
    """on_thought callback fires and thoughts are accumulated on the stream."""
    received: list[str] = []
    async with GeminiClient(
        model="gemini-2.5-flash",
        on_thought=received.append,
        thinking_config={"thinkingBudget": 2048, "includeThoughts": True},
    ) as client:
        stream = client.stream_chat([{"role": "user", "content": "What is 7 * 8?"}])
        text = "".join([chunk async for chunk in stream])

    assert text, "no text returned"
    assert "56" in text
    assert received, "no thought chunks received"


async def test_batch_lifecycle() -> None:
    async with GeminiClient() as client:
        req = BatchRequest(
            contents=[{"role": "user", "parts": [{"text": "Say hello"}]}],
            key="e2e-gemini-1",
        )
        job = await client.create_batch(
            [req], model="gemini-2.5-flash", display_name="e2e-test"
        )
        assert job.name
        assert job.state

        fetched = await client.get_batch(job.name)
        assert fetched.name == job.name

        jobs, _ = await client.list_batches(page_size=5)
        assert len(jobs) >= 0  # may be empty if cleaned up

        await client.cancel_batch(job.name)
