import pytest

from padwan_llm import OpenAIClient
from padwan_llm.openai import BatchRequest

from .conftest import skip_no_openai

pytestmark = [pytest.mark.e2e, skip_no_openai]


async def test_batch_lifecycle() -> None:
    async with OpenAIClient() as client:
        req = BatchRequest(
            body={"messages": [{"role": "user", "content": "Say hello"}]},
            custom_id="e2e-openai-1",
        )
        job = await client.create_batch([req], model="gpt-4o-mini")
        assert job.id
        assert not job.is_terminal

        fetched = await client.get_batch(job.id)
        assert fetched.id == job.id

        jobs, _ = await client.list_batches(limit=5)
        assert any(j.id == job.id for j in jobs)

        cancelled = await client.cancel_batch(job.id)
        assert cancelled.id == job.id
