import asyncio

import pytest

from padwan_llm import GrokClient
from padwan_llm.grok import GrokBatchRequest

from .conftest import skip_no_grok

pytestmark = [pytest.mark.e2e, skip_no_grok]


async def test_batch_lifecycle() -> None:
    async with GrokClient() as client:
        req = GrokBatchRequest(
            body={"messages": [{"role": "user", "content": "Say hello"}]},
            custom_id="e2e-grok-1",
        )
        job = await client.create_batch(
            [req], name="e2e-test", model="grok-4-1-fast-non-reasoning"
        )
        assert job.batch_id
        assert job.num_requests == 1

        fetched = await client.get_batch(job.batch_id)
        assert fetched.batch_id == job.batch_id

        jobs, _ = await client.list_batches(limit=5)
        assert any(j.batch_id == job.batch_id for j in jobs)

        cancelled = await client.cancel_batch(job.batch_id)
        assert cancelled.batch_id == job.batch_id


async def test_batch_completion() -> None:
    """Submit a small batch and poll until results are available."""
    async with GrokClient() as client:
        req = GrokBatchRequest(
            body={"messages": [{"role": "user", "content": "Reply with only 'hi'"}]},
            custom_id="e2e-grok-poll",
        )
        job = await client.create_batch(
            [req], name="e2e-poll", model="grok-4-1-fast-non-reasoning"
        )

        # Poll with exponential backoff, max ~5 min
        delay = 2.0
        for _ in range(15):
            await asyncio.sleep(delay)
            job = await client.get_batch(job.batch_id)
            if job.is_terminal:
                break
            delay = min(delay * 1.5, 30.0)
        else:
            pytest.skip("Batch did not complete within timeout")

        assert job.succeeded, f"Batch failed: {job}"

        results, _ = await client.get_batch_results(job.batch_id)
        assert len(results) == 1
        assert results[0].custom_id == "e2e-grok-poll"
        assert results[0].content
        assert results[0].total_tokens > 0
