import base64
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from .._base import NO_TURN_DETECTION, RealtimeClientBase
from .._ws import READ_POLL_INTERVAL, WsConnection, enable_read_polling
from ..errors import LLMError
from ..logs import log
from .client import _GeminiAuth

if TYPE_CHECKING:
    import niquests

__all__ = (
    "DEFAULT_LIVE_MODEL",
    "LIVE_ENDPOINT",
    "LIVE_INPUT_SAMPLE_RATE",
    "LIVE_OUTPUT_SAMPLE_RATE",
    "NO_TURN_DETECTION",
    "GeminiLiveModel",
    "GeminiRealtimeClient",
    "GeminiRealtimeConnection",
    "GeminiVoice",
)

LIVE_ENDPOINT = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_LIVE_MODEL = "gemini-3.1-flash-live-preview"

GeminiLiveModel = Literal["gemini-3.1-flash-live-preview"]

# The Live API is asymmetric: mono little-endian PCM16 at 16 kHz in, 24 kHz out.
LIVE_INPUT_SAMPLE_RATE = 16_000
LIVE_OUTPUT_SAMPLE_RATE = 24_000

_KnownGeminiVoice = Literal[
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Aoede",
    "Leda",
    "Orus",
    "Zephyr",
]
# `| str` keeps arbitrary voices valid while the Literal member drives IDE completion.
type GeminiVoice = _KnownGeminiVoice | str
_GEMINI_VOICES: tuple[str, ...] = get_args(_KnownGeminiVoice)


@dataclass
class GeminiRealtimeClient(_GeminiAuth, RealtimeClientBase["GeminiRealtimeConnection"]):
    """Speech-to-speech client for the Gemini Live API over a WebSocket.

    Server VAD by default; pass :data:`NO_TURN_DETECTION` to mark turns
    manually. Wire shapes: https://ai.google.dev/api/live
    """

    model: str = DEFAULT_LIVE_MODEL
    base_url: str = LIVE_ENDPOINT
    instructions: str | None = None
    voice: GeminiVoice = "Puck"
    turn_detection: Mapping[str, Any] | str | None = None
    transcription: bool = True
    output_modalities: Sequence[str] = ("audio",)

    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        # The Live API authenticates via the ``key`` query parameter on the
        # upgrade request (see _connect), not via headers.
        pass

    def setup_payload(self) -> dict[str, Any]:
        """Build the ``BidiGenerateContentSetup`` message opening the session.

        Split out from :meth:`_connect` so the payload can be inspected and
        unit-tested without a live connection.
        """
        generation_config: dict[str, Any] = {
            "responseModalities": [m.upper() for m in self.output_modalities],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self.voice}}
            },
        }
        setup: dict[str, Any] = {
            "model": f"models/{self.model}",
            "generationConfig": generation_config,
        }
        if self.instructions:
            setup["systemInstruction"] = {"parts": [{"text": self.instructions}]}
        if isinstance(self.turn_detection, str):  # NO_TURN_DETECTION
            setup["realtimeInputConfig"] = {
                "automaticActivityDetection": {"disabled": True}
            }
        elif self.turn_detection:
            setup["realtimeInputConfig"] = {
                "automaticActivityDetection": dict(self.turn_detection)
            }
        if self.transcription:
            setup["inputAudioTranscription"] = {}
            setup["outputAudioTranscription"] = {}
        return {"setup": setup}

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator["GeminiRealtimeConnection"]:
        # timeout bounds only the upgrade handshake; afterwards the read
        # timeout is re-armed as the poll interval (see enable_read_polling).
        resp = await self.session.get(
            self.base_url,
            params={"key": self._api_key},
            timeout=self.timeout,
        )
        ext = resp.extension
        if ext is None:
            raise LLMError(
                "gemini",
                f"live handshake did not upgrade to a websocket "
                f"(status {resp.status_code})",
            )
        enable_read_polling(ext, READ_POLL_INTERVAL)
        conn = GeminiRealtimeConnection(ext)
        await conn.send_event(self.setup_payload())
        async for message in conn:
            if "setupComplete" in message:
                break
            log.debug("ignoring pre-setup live message: %s", message)
        else:
            raise LLMError("gemini", "live session closed before setupComplete")
        try:
            yield conn
        finally:
            await conn.close()


@dataclass
class GeminiRealtimeConnection(WsConnection):
    """An open Gemini Live session.

    Stream microphone audio with :meth:`append_audio` (mono PCM16 at 16 kHz)
    and consume server messages by async-iterating the connection; model audio
    in :meth:`audio_delta_bytes` is PCM16 at 24 kHz.
    """

    async def append_audio(self, pcm16: bytes) -> None:
        """Stream a chunk of mono PCM16 @ 16 kHz microphone audio."""
        audio = base64.b64encode(pcm16).decode("ascii")
        await self.send_event(
            {
                "realtimeInput": {
                    "audio": {
                        "data": audio,
                        "mimeType": f"audio/pcm;rate={LIVE_INPUT_SAMPLE_RATE}",
                    }
                }
            }
        )

    async def send_text(self, text: str) -> None:
        """Send a text turn instead of audio."""
        await self.send_event({"realtimeInput": {"text": text}})

    async def activity_start(self) -> None:
        """Mark the start of a user turn (only needed without automatic VAD)."""
        await self.send_event({"realtimeInput": {"activityStart": {}}})

    async def activity_end(self) -> None:
        """Mark the end of a user turn (only needed without automatic VAD)."""
        await self.send_event({"realtimeInput": {"activityEnd": {}}})

    async def audio_stream_end(self) -> None:
        """Signal that the audio stream is paused (e.g. microphone muted)."""
        await self.send_event({"realtimeInput": {"audioStreamEnd": True}})

    @staticmethod
    def audio_delta_bytes(message: Mapping[str, Any]) -> bytes | None:
        """Decode the PCM16 audio parts of a ``serverContent.modelTurn`` message."""
        parts = message.get("serverContent", {}).get("modelTurn", {}).get("parts")
        if not parts:
            return None
        chunks = [
            base64.b64decode(part["inlineData"]["data"])
            for part in parts
            if isinstance(part.get("inlineData", {}).get("data"), str)
        ]
        return b"".join(chunks) or None

    @staticmethod
    def is_turn_complete(message: Mapping[str, Any]) -> bool:
        """True when the model has finished its turn."""
        return bool(message.get("serverContent", {}).get("turnComplete"))

    @staticmethod
    def input_transcript(message: Mapping[str, Any]) -> str | None:
        """Transcription text of your own speech, if present."""
        return (
            message.get("serverContent", {}).get("inputTranscription", {}).get("text")
        )

    @staticmethod
    def output_transcript(message: Mapping[str, Any]) -> str | None:
        """Transcription text of the model's speech, if present."""
        return (
            message.get("serverContent", {}).get("outputTranscription", {}).get("text")
        )
