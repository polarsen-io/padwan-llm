import base64

import pytest

from padwan_llm._json import loads as _json_loads
from padwan_llm.errors import LLMError
from padwan_llm.openai.realtime import (
    RealtimeClient,
    RealtimeConnection,
    RealtimeServerEvent,
)


class FakeExt:
    """Stand-in for niquests' websocket extension."""

    def __init__(self, incoming: list[str | bytes | None] | None = None):
        self.sent: list[str] = []
        self._incoming = list(incoming or [])
        self.closed = False

    async def send_payload(self, buf: str | bytes) -> None:
        self.sent.append(buf if isinstance(buf, str) else buf.decode())

    async def next_payload(self) -> str | bytes | None:
        return self._incoming.pop(0) if self._incoming else None

    async def close(self) -> None:
        self.closed = True


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        RealtimeClient()
    # Explicit key bypasses the environment lookup.
    assert RealtimeClient(api_key="sk-test").model == "gpt-realtime"


@pytest.mark.parametrize(
    "kwargs, expected_turn, has_transcription, instructions",
    [
        pytest.param(
            {},
            {"type": "server_vad"},
            True,
            None,
            id="defaults-server-vad",
        ),
        pytest.param(
            {"turn_detection": {"type": "semantic_vad"}},
            {"type": "semantic_vad"},
            True,
            None,
            id="semantic-vad-override",
        ),
        pytest.param(
            {"turn_detection": {}},
            {"type": "server_vad"},
            True,
            None,
            id="empty-mapping-falls-back",
        ),
        pytest.param(
            {"transcription_model": None, "instructions": "Parla italiano."},
            {"type": "server_vad"},
            False,
            "Parla italiano.",
            id="no-transcription-with-instructions",
        ),
    ],
)
def test_session_payload(
    kwargs, expected_turn, has_transcription, instructions
) -> None:
    conn = RealtimeConnection(FakeExt())
    payload = conn.session_payload(
        instructions=kwargs.get("instructions"),
        voice="cedar",
        turn_detection=kwargs.get("turn_detection"),
        transcription_model=kwargs.get("transcription_model", "whisper-1"),
        output_modalities=("audio",),
    )
    assert payload["type"] == "session.update"
    session = payload["session"]
    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    audio = session["audio"]
    assert audio["input"]["format"] == {"type": "audio/pcm", "rate": 24_000}
    assert audio["output"] == {
        "format": {"type": "audio/pcm", "rate": 24_000},
        "voice": "cedar",
    }
    assert audio["input"]["turn_detection"] == expected_turn
    assert ("transcription" in audio["input"]) is has_transcription
    assert session.get("instructions") == instructions


async def test_append_audio_base64_roundtrip() -> None:
    ext = FakeExt()
    conn = RealtimeConnection(ext)
    pcm = b"\x01\x02\x03\x04" * 16
    await conn.append_audio(pcm)
    event = _json_loads(ext.sent[-1])
    assert event["type"] == "input_audio_buffer.append"
    assert base64.b64decode(event["audio"]) == pcm


@pytest.mark.parametrize(
    "event, expected",
    [
        pytest.param(
            {
                "type": RealtimeServerEvent.AUDIO_DELTA,
                "delta": base64.b64encode(b"abcd").decode(),
            },
            b"abcd",
            id="audio-delta-decoded",
        ),
        pytest.param(
            {"type": RealtimeServerEvent.AUDIO_TRANSCRIPT_DELTA, "delta": "ciao"},
            None,
            id="non-audio-event-ignored",
        ),
        pytest.param(
            {"type": RealtimeServerEvent.AUDIO_DELTA},
            None,
            id="missing-delta",
        ),
    ],
)
def test_audio_delta_bytes(event, expected) -> None:
    assert RealtimeConnection.audio_delta_bytes(event) == expected


async def test_aiter_parses_until_none() -> None:
    incoming = [
        '{"type": "session.created"}',
        b'{"type": "response.output_audio.delta", "delta": "AAAA"}',
        None,  # close
        '{"type": "never.seen"}',
    ]
    conn = RealtimeConnection(FakeExt(incoming))
    seen = [event["type"] async for event in conn]
    assert seen == ["session.created", "response.output_audio.delta"]
