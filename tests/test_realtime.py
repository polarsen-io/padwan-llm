import asyncio
import base64
from types import SimpleNamespace
from typing import cast

import niquests
import pytest

from padwan_llm._json import loads as _json_loads
from padwan_llm.errors import LLMError
from padwan_llm.openai.realtime import (
    NO_TURN_DETECTION,
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


class LockedExt(FakeExt):
    """FakeExt mimicking the transport: one lock on the socket, timed-out reads.

    ``next_payload`` holds the lock while parked (like the real traffic police)
    and raises ``ReadTimeout`` after the poll interval, releasing the lock so a
    queued ``send_payload`` can borrow the socket.
    """

    def __init__(self):
        super().__init__()
        self.lock = asyncio.Lock()
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def next_payload(self) -> str | bytes | None:
        async with self.lock:
            try:
                return await asyncio.wait_for(self.queue.get(), 0.05)
            except TimeoutError:
                raise niquests.exceptions.ReadTimeout("poll tick")

    async def send_payload(self, buf: str | bytes) -> None:
        async with self.lock:
            await super().send_payload(buf)


async def test_send_progresses_while_read_is_parked() -> None:
    ext = LockedExt()
    conn = RealtimeConnection(ext)
    received: list[dict] = []

    async def consume() -> None:
        async for event in conn:
            received.append(event)

    consumer = asyncio.create_task(consume())
    try:
        await asyncio.sleep(0.01)  # reader is parked, holding the socket lock

        # Must go out at the next poll tick, not wait for a server event.
        await asyncio.wait_for(conn.send_event({"type": "input_audio_buffer.clear"}), 1)
        assert [_json_loads(s) for s in ext.sent] == [
            {"type": "input_audio_buffer.clear"}
        ]

        # Poll ticks are swallowed: later events still come through.
        await ext.queue.put('{"type":"session.created"}')
        await ext.queue.put(None)
        await asyncio.wait_for(consumer, 1)
        assert received == [{"type": "session.created"}]
    finally:
        await conn.close()
        consumer.cancel()


class FakeSession:
    """Stand-in for niquests.AsyncSession capturing the handshake request."""

    def __init__(self, ext: FakeExt | None = None):
        self._ext = ext
        self.closed = False
        self.get_kwargs: dict | None = None

    async def get(self, url, **kwargs):
        self.get_kwargs = {"url": url, **kwargs}
        return SimpleNamespace(extension=self._ext, status_code=101)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True


async def test_connect_with_caller_session_leaves_it_open() -> None:
    ext = FakeExt()
    fake = FakeSession(ext)
    client = RealtimeClient(api_key="sk-test")
    async with client.connect(session=cast("niquests.AsyncSession", fake)):
        pass
    assert ext.closed is True
    assert fake.closed is False
    assert fake.get_kwargs is not None
    assert fake.get_kwargs["params"] == {"model": "gpt-realtime"}


async def test_connect_forwards_session_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext = FakeExt()
    fake = FakeSession(ext)
    captured: dict = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(niquests, "AsyncSession", factory)
    client = RealtimeClient(api_key="sk-test")
    async with client.connect(session_kwargs={"happy_eyeballs": True}):
        pass
    assert captured == {"happy_eyeballs": True}
    assert fake.closed is True  # internally created sessions are closed on exit


async def test_connect_rejects_session_and_session_kwargs() -> None:
    client = RealtimeClient(api_key="sk-test")
    with pytest.raises(ValueError, match="not both"):
        async with client.connect(
            session=cast("niquests.AsyncSession", FakeSession()), session_kwargs={}
        ):
            pass


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
            {"turn_detection": NO_TURN_DETECTION},
            None,
            True,
            None,
            id="disabled-emits-null",
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
