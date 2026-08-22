import pytest

from padwan_llm.errors import LLMError
from padwan_llm.mistral.client import MistralClient


def test_prepare_messages_converts_audio_parts():
    client = MistralClient(api_key="k")
    messages = [
        {"role": "system", "content": "be brief"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe this"},
                {
                    "type": "input_audio",
                    "input_audio": {"data": "UklGRg==", "format": "wav"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]

    prepared = client._prepare_messages(messages)

    assert prepared[0] is messages[0]
    assert prepared[1]["content"] == [
        {"type": "text", "text": "transcribe this"},
        {"type": "input_audio", "input_audio": "UklGRg=="},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


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
