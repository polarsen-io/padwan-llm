import array
import asyncio
import wave

import pytest

from padwan_llm import RealtimeClient
from padwan_llm.gemini.realtime import GeminiRealtimeConnection
from padwan_llm.openai.realtime import RealtimeConnection, RealtimeServerEvent

from .conftest import AUDIO_FIXTURE, skip_no_gemini, skip_no_grok, skip_no_openai

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not AUDIO_FIXTURE.exists(), reason="audio fixture not found"),
]

INSTRUCTIONS = "You are a test agent. Reply with one short spoken sentence."
TIMEOUT = 90.0
CHUNK = 8_000  # ~0.25 s of PCM16 @ 16 kHz per append


def _fixture_pcm() -> bytes:
    """The fixture's raw frames: mono little-endian PCM16 @ 16 kHz speech."""
    with wave.open(str(AUDIO_FIXTURE)) as w:
        return w.readframes(w.getnframes())


def _upsample_16k_to_24k(pcm: bytes) -> bytes:
    """Linear-interpolate 16 kHz PCM16 to the 24 kHz the OpenAI API expects."""
    src = array.array("h")
    src.frombytes(pcm)
    out = array.array("h")
    for j in range(len(src) * 3 // 2):
        pos = j * 2 / 3
        i = int(pos)
        a = src[i]
        b = src[min(i + 1, len(src) - 1)]
        out.append(int(a + (b - a) * (pos - i)))
    return out.tobytes()


def _silence(seconds: float, rate: int) -> bytes:
    """Trailing silence so server VAD detects the end of the turn."""
    return b"\x00\x00" * int(seconds * rate)


async def _collect_response(conn: RealtimeConnection, pcm: bytes) -> bytes:
    """Stream *pcm*, then iterate events until a response completes.

    Pauses inside the clip can make VAD split it into several turns; a new turn
    cancels the in-flight response ("turn_detected"), so only a ``completed``
    response ends the wait.
    """
    for i in range(0, len(pcm), CHUNK):
        await conn.append_audio(pcm[i : i + CHUNK])
    audio = bytearray()
    async for event in conn:
        if event.get("type") == RealtimeServerEvent.ERROR:
            pytest.fail(f"server error event: {event}")
        if delta := conn.audio_delta_bytes(event):
            audio.extend(delta)
        if (
            event.get("type") == RealtimeServerEvent.RESPONSE_DONE
            and event.get("response", {}).get("status") == "completed"
        ):
            break
    return bytes(audio)


@skip_no_openai
async def test_openai_realtime_speech_roundtrip() -> None:
    pcm = _upsample_16k_to_24k(_fixture_pcm()) + _silence(1.0, 24_000)
    async with asyncio.timeout(TIMEOUT):
        async with RealtimeClient(instructions=INSTRUCTIONS) as conn:
            audio = await _collect_response(conn, pcm)
    assert len(audio) > 4_800, "no audible response received"


@skip_no_grok
async def test_grok_realtime_speech_roundtrip() -> None:
    pcm = _fixture_pcm() + _silence(1.0, 16_000)
    async with asyncio.timeout(TIMEOUT):
        async with RealtimeClient(
            "grok-voice-latest", instructions=INSTRUCTIONS, sample_rate=16_000
        ) as conn:
            audio = await _collect_response(conn, pcm)
    assert len(audio) > 3_200, "no audible response received"


@skip_no_gemini
async def test_gemini_live_speech_roundtrip() -> None:
    pcm = _fixture_pcm() + _silence(1.0, 16_000)
    async with asyncio.timeout(TIMEOUT):
        async with RealtimeClient(
            "gemini-3.1-flash-live-preview", instructions=INSTRUCTIONS
        ) as conn:
            for i in range(0, len(pcm), CHUNK):
                await conn.append_audio(pcm[i : i + CHUNK])
            audio = bytearray()
            async for message in conn:
                if delta := GeminiRealtimeConnection.audio_delta_bytes(message):
                    audio.extend(delta)
                if GeminiRealtimeConnection.is_turn_complete(message):
                    break
    assert len(audio) > 4_800, "no audible response received"
