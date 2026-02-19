import pytest

from padwan_llm.errors import LLMError
from padwan_llm.mistral.client import MistralClient


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no-source"),
        pytest.param({"file": b"audio", "file_id": "id"}, id="two-sources"),
        pytest.param({"file": b"audio", "file_url": "http://x"}, id="file-and-url"),
    ],
)
async def test_transcribe_source_validation(kwargs: dict):
    client = MistralClient(api_key="k")
    async with client:
        with pytest.raises(LLMError, match="exactly one"):
            await client.transcribe(**kwargs)
