import asyncio
import base64
import re
from typing import get_type_hints

import niquests
import pytest
from google.genai.types import (
    AutomaticActivityDetectionDict,
    LiveClientRealtimeInputDict,
    LiveClientSetupDict,
    LiveConnectConfigDict,
    LiveServerContentDict,
)

from padwan_llm import (
    GeminiRealtimeClient,
    GeminiRealtimeConnection,
    GrokRealtimeClient,
    OpenAIRealtimeClient,
    RealtimeClient,
)
from padwan_llm._json import loads as _json_loads
from padwan_llm.errors import LLMError
from padwan_llm.gemini.models import (
    AutomaticActivityDetection,
    LiveGenerationConfig,
    LiveSetup,
)
from padwan_llm.gemini.realtime import DEFAULT_LIVE_MODEL, LIVE_ENDPOINT
from padwan_llm.openai.realtime import (
    NO_TURN_DETECTION,
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


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        RealtimeClient()
    # Explicit key bypasses the environment lookup.
    assert RealtimeClient(api_key="sk-test").model == "gpt-realtime"


@pytest.mark.parametrize(
    "api_key, padwan_key, base_url, expected_key, expected_url",
    [
        pytest.param(
            "sk-explicit",
            "pk-gateway",
            "wss://example.com/rt",
            "sk-explicit",
            "wss://example.com/rt",
            id="explicit_key_wins_over_gateway",
        ),
        pytest.param(
            None,
            "pk-gateway",
            "wss://example.com/rt",
            "pk-gateway",
            "wss://example.com/rt",
            id="custom_url_prefers_padwan_key",
        ),
        pytest.param(
            None,
            "pk-gateway",
            None,
            "sk-env",
            "wss://api.openai.com/v1/realtime",
            id="default_url_ignores_padwan_key",
        ),
        pytest.param(
            None,
            None,
            "wss://example.com/rt",
            "sk-env",
            "wss://example.com/rt",
            id="custom_url_falls_back_to_provider_env",
        ),
    ],
)
def test_factory_key_and_url_resolution(
    monkeypatch: pytest.MonkeyPatch,
    api_key: str | None,
    padwan_key: str | None,
    base_url: str | None,
    expected_key: str,
    expected_url: str,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    if padwan_key is None:
        monkeypatch.delenv("PADWAN_API_KEY", raising=False)
    else:
        monkeypatch.setenv("PADWAN_API_KEY", padwan_key)
    client = (
        RealtimeClient(api_key=api_key, base_url=base_url)
        if base_url
        else RealtimeClient(api_key=api_key)
    )
    assert client._api_key == expected_key
    assert client.base_url == expected_url


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


@pytest.mark.parametrize(
    "kwargs, expected_vad, has_transcription, instructions",
    [
        pytest.param({}, None, True, None, id="defaults-automatic-vad"),
        pytest.param(
            {"turn_detection": NO_TURN_DETECTION},
            {"disabled": True},
            True,
            None,
            id="disabled-manual-turns",
        ),
        pytest.param(
            {"turn_detection": {"silenceDurationMs": 500}},
            {"silenceDurationMs": 500},
            True,
            None,
            id="mapping-verbatim",
        ),
        pytest.param(
            {"transcription": False, "instructions": "Parla italiano."},
            None,
            False,
            "Parla italiano.",
            id="no-transcription-with-instructions",
        ),
    ],
)
def test_gemini_setup_payload(
    kwargs, expected_vad, has_transcription, instructions
) -> None:
    client = GeminiRealtimeClient(api_key="g-test", voice="Kore", **kwargs)
    payload = client.setup_payload()
    setup = payload["setup"]
    assert setup["model"] == f"models/{DEFAULT_LIVE_MODEL}"
    config = setup["generationConfig"]
    assert config["responseModalities"] == ["AUDIO"]
    voice_name = config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]
    assert voice_name == {"voiceName": "Kore"}
    vad = setup.get("realtimeInputConfig", {}).get("automaticActivityDetection")
    assert vad == expected_vad
    assert ("inputAudioTranscription" in setup) is has_transcription
    assert ("outputAudioTranscription" in setup) is has_transcription
    if instructions:
        assert setup.get("systemInstruction") == {"parts": [{"text": instructions}]}
    else:
        assert "systemInstruction" not in setup


async def test_gemini_append_audio_16k_mime() -> None:
    ext = FakeExt()
    conn = GeminiRealtimeConnection(ext)
    pcm = b"\x01\x02" * 32
    await conn.append_audio(pcm)
    event = _json_loads(ext.sent[-1])
    audio = event["realtimeInput"]["audio"]
    assert audio["mimeType"] == "audio/pcm;rate=16000"
    assert base64.b64decode(audio["data"]) == pcm


@pytest.mark.parametrize(
    "message, expected",
    [
        pytest.param(
            {
                "serverContent": {
                    "modelTurn": {
                        "parts": [
                            {"inlineData": {"data": base64.b64encode(b"ab").decode()}},
                            {"text": "ignored"},
                            {"inlineData": {"data": base64.b64encode(b"cd").decode()}},
                        ]
                    }
                }
            },
            b"abcd",
            id="audio-parts-joined",
        ),
        pytest.param({"serverContent": {"turnComplete": True}}, None, id="no-audio"),
        pytest.param({"setupComplete": {}}, None, id="setup-complete"),
    ],
)
def test_gemini_audio_delta_bytes(message, expected) -> None:
    assert GeminiRealtimeConnection.audio_delta_bytes(message) == expected


@pytest.mark.parametrize(
    "model, expected_cls, env_var, expected_url",
    [
        pytest.param(
            "gpt-realtime",
            OpenAIRealtimeClient,
            "OPENAI_API_KEY",
            "wss://api.openai.com/v1/realtime",
            id="openai",
        ),
        pytest.param(
            "gemini-3.1-flash-live-preview",
            GeminiRealtimeClient,
            "GEMINI_API_KEY",
            LIVE_ENDPOINT,
            id="gemini",
        ),
        pytest.param(
            "grok-voice-latest",
            GrokRealtimeClient,
            "GROK_API_KEY",
            "wss://api.x.ai/v1/realtime",
            id="grok",
        ),
    ],
)
def test_factory_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    expected_cls: type,
    env_var: str,
    expected_url: str,
) -> None:
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GROK_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv(env_var, "sk-env")
    client = RealtimeClient(model)
    assert type(client) is expected_cls
    assert client._api_key == "sk-env"
    assert client.base_url == expected_url


def test_factory_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # voice=None keeps each provider's default; explicit voice overrides it.
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("GROK_API_KEY", "x")
    assert RealtimeClient("gemini-3.1-flash-live-preview").voice == "Puck"
    assert RealtimeClient("grok-voice-latest").voice == "eve"
    assert RealtimeClient("grok-voice-latest", voice="ara").voice == "ara"
    # Grok transcribes natively: the OpenAI-style default is not forwarded,
    # and an explicit transcription model is rejected.
    assert RealtimeClient("grok-voice-latest").transcription_model is None
    with pytest.raises(ValueError, match="transcribes natively"):
        RealtimeClient("grok-voice-latest", transcription_model="scribe-x")
    with pytest.raises(ValueError, match="sample_rate is fixed"):
        RealtimeClient("gemini-3.1-flash-live-preview", sample_rate=8_000)


def _camel_to_snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


@pytest.mark.parametrize(
    "local_type, sdk_types",
    [
        # The SDK splits the wire setup across LiveClientSetup and
        # LiveConnectConfig (its converters merge the latter into ``setup``).
        pytest.param(
            LiveSetup, (LiveClientSetupDict, LiveConnectConfigDict), id="LiveSetup"
        ),
        pytest.param(
            LiveGenerationConfig, (LiveConnectConfigDict,), id="LiveGenerationConfig"
        ),
        pytest.param(
            AutomaticActivityDetection,
            (AutomaticActivityDetectionDict,),
            id="AutomaticActivityDetection",
        ),
    ],
)
def test_gemini_live_sdk_compat(local_type: type, sdk_types: tuple[type, ...]) -> None:
    """Every field of our local Live types maps to a field in google-genai's."""
    sdk_fields = {field for t in sdk_types for field in get_type_hints(t)}
    for key in get_type_hints(local_type):
        assert _camel_to_snake(key) in sdk_fields, f"{local_type.__name__}.{key}"


async def test_gemini_realtime_input_sdk_compat() -> None:
    """Every realtimeInput key we send maps to a LiveClientRealtimeInput field."""
    ext = FakeExt()
    conn = GeminiRealtimeConnection(ext)
    await conn.append_audio(b"\x00\x00")
    await conn.send_text("hi")
    await conn.activity_start()
    await conn.activity_end()
    await conn.audio_stream_end()
    fields = set(get_type_hints(LiveClientRealtimeInputDict))
    for sent in ext.sent:
        for key in _json_loads(sent)["realtimeInput"]:
            assert _camel_to_snake(key) in fields, key


@pytest.mark.parametrize(
    "key",
    [
        pytest.param("modelTurn", id="model_turn"),
        pytest.param("turnComplete", id="turn_complete"),
        pytest.param("inputTranscription", id="input_transcription"),
        pytest.param("outputTranscription", id="output_transcription"),
    ],
)
def test_gemini_server_content_sdk_compat(key: str) -> None:
    """The serverContent keys our accessors read exist in LiveServerContent."""
    assert _camel_to_snake(key) in get_type_hints(LiveServerContentDict)
