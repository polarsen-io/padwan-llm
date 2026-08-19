from __future__ import annotations

import base64
import enum
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_args

from .._base import NO_TURN_DETECTION, RealtimeClientBase
from .._ws import READ_POLL_INTERVAL, WsConnection, enable_read_polling
from ..errors import LLMError
from .client import _OpenAIAuth

if TYPE_CHECKING:
    import niquests
    from openai.types.realtime import (
        RealtimeAudioConfigInputParam,
        RealtimeAudioInputTurnDetectionParam,
        RealtimeSessionCreateRequestParam,
        SessionUpdateEventParam,
    )
    from openai.types.realtime.realtime_audio_formats_param import AudioPCM

__all__ = (
    "DEFAULT_REALTIME_MODEL",
    "NO_TURN_DETECTION",
    "REALTIME_ENDPOINT",
    "REALTIME_SAMPLE_RATE",
    "OpenAIRealtimeClient",
    "RealtimeConnection",
    "RealtimeServerEvent",
    "RealtimeVoice",
)

REALTIME_ENDPOINT = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_MODEL = "gpt-realtime"

# gpt-realtime exchanges mono little-endian PCM16 at 24 kHz.
REALTIME_SAMPLE_RATE = 24_000


# "marin" and "cedar" are the voices introduced with gpt-realtime; the rest carry
# over from the preview models. Checked against the SDK by bin/drift.
_KnownVoice = Literal[
    "marin",
    "cedar",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
]
# `| str` keeps arbitrary voices valid (the SDK types voice the same way) while
# the Literal member drives IDE completion.
type RealtimeVoice = _KnownVoice | str
_VOICES: tuple[str, ...] = get_args(_KnownVoice)


class RealtimeServerEvent(enum.StrEnum):
    """Server event ``type`` strings emitted by the GA Realtime API."""

    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SPEECH_STARTED = "input_audio_buffer.speech_started"
    SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    AUDIO_DELTA = "response.output_audio.delta"
    AUDIO_DONE = "response.output_audio.done"
    AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
    INPUT_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
    INPUT_TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    ERROR = "error"


@dataclass
class OpenAIRealtimeClient(_OpenAIAuth, RealtimeClientBase["RealtimeConnection"]):
    """Speech-to-speech client for the OpenAI Realtime API over a WebSocket.

    Server VAD by default; pass :data:`NO_TURN_DETECTION` to drive turns
    manually. Wire shapes: https://platform.openai.com/docs/guides/realtime
    """

    model: str = DEFAULT_REALTIME_MODEL
    base_url: str = REALTIME_ENDPOINT
    instructions: str | None = None
    voice: RealtimeVoice = "marin"
    turn_detection: Mapping[str, Any] | str | None = None
    transcription_model: str | None = "whisper-1"
    output_modalities: Sequence[str] = ("audio",)
    sample_rate: int = REALTIME_SAMPLE_RATE

    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        session.headers["Authorization"] = f"Bearer {self._api_key}"

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[RealtimeConnection]:
        # timeout bounds only the upgrade handshake; afterwards the read
        # timeout is re-armed as the poll interval (see _enable_read_polling).
        resp = await self.session.get(
            self.base_url,
            params={"model": self.model},
            timeout=self.timeout,
        )
        ext = resp.extension
        if ext is None:
            raise LLMError(
                "openai",
                f"realtime handshake did not upgrade to a websocket "
                f"(status {resp.status_code})",
            )
        enable_read_polling(ext, READ_POLL_INTERVAL)
        conn = RealtimeConnection(ext, sample_rate=self.sample_rate)
        await conn.configure(
            instructions=self.instructions,
            voice=self.voice,
            turn_detection=self.turn_detection,
            transcription_model=self.transcription_model,
            output_modalities=list(self.output_modalities),
        )
        try:
            yield conn
        finally:
            await conn.close()


@dataclass
class RealtimeConnection(WsConnection):
    """An open realtime session.

    Stream microphone audio with :meth:`append_audio` and consume server events by
    async-iterating the connection. Audio is mono little-endian PCM16 at
    *sample_rate* Hz in both directions.
    """

    sample_rate: int = REALTIME_SAMPLE_RATE

    def session_payload(
        self,
        *,
        instructions: str | None,
        voice: RealtimeVoice,
        turn_detection: Mapping[str, Any] | str | None,
        transcription_model: str | None,
        output_modalities: Sequence[str],
    ) -> dict[str, Any]:
        """Build the ``session.update`` event for the GA Realtime schema.

        Split out from :meth:`configure` so the payload can be inspected and
        unit-tested without a live connection; the construction is typed against
        the OpenAI SDK's ``session.update`` params so wire drift surfaces at
        type-check time on SDK bumps. *turn_detection* of
        :data:`NO_TURN_DETECTION` emits a JSON ``null`` to disable server VAD; a
        falsy value uses the ``server_vad`` default; a mapping is sent verbatim.
        """
        audio_format: AudioPCM = {
            "type": "audio/pcm",
            # The SDK pins 24 kHz; Grok speaks the same wire at other rates.
            "rate": cast("Literal[24000]", self.sample_rate),
        }
        if isinstance(turn_detection, str):
            detection: RealtimeAudioInputTurnDetectionParam | None = None
        elif turn_detection:
            # Caller-supplied mapping, sent verbatim.
            detection = cast(
                "RealtimeAudioInputTurnDetectionParam", dict(turn_detection)
            )
        else:
            detection = {"type": "server_vad"}
        audio_input: RealtimeAudioConfigInputParam = {
            "format": audio_format,
            "turn_detection": detection,
        }
        if transcription_model:
            audio_input["transcription"] = {"model": transcription_model}
        session: RealtimeSessionCreateRequestParam = {
            "type": "realtime",
            "output_modalities": cast(
                "list[Literal['text', 'audio']]", list(output_modalities)
            ),
            "audio": {
                "input": audio_input,
                "output": {"format": audio_format, "voice": voice},
            },
        }
        if instructions:
            session["instructions"] = instructions
        event: SessionUpdateEventParam = {"type": "session.update", "session": session}
        return cast("dict[str, Any]", event)

    async def configure(
        self,
        *,
        instructions: str | None = None,
        voice: RealtimeVoice = "marin",
        turn_detection: Mapping[str, Any] | str | None = None,
        transcription_model: str | None = "whisper-1",
        output_modalities: Sequence[str] = ("audio",),
    ) -> None:
        """Send a ``session.update`` reconfiguring the live session."""
        await self.send_event(
            self.session_payload(
                instructions=instructions,
                voice=voice,
                turn_detection=turn_detection,
                transcription_model=transcription_model,
                output_modalities=output_modalities,
            )
        )

    async def append_audio(self, pcm16: bytes) -> None:
        """Append a chunk of mono PCM16 audio to the input buffer."""
        audio = base64.b64encode(pcm16).decode("ascii")
        await self.send_event({"type": "input_audio_buffer.append", "audio": audio})

    async def commit_audio(self) -> None:
        """Commit the input buffer as a turn (only needed without server VAD)."""
        await self.send_event({"type": "input_audio_buffer.commit"})

    async def create_response(self) -> None:
        """Ask the model to respond now (only needed without server VAD)."""
        await self.send_event({"type": "response.create"})

    @staticmethod
    def audio_delta_bytes(event: Mapping[str, Any]) -> bytes | None:
        """Decode the PCM16 payload of a ``response.output_audio.delta`` event."""
        if event.get("type") != RealtimeServerEvent.AUDIO_DELTA:
            return None
        delta = event.get("delta")
        return base64.b64decode(delta) if isinstance(delta, str) else None
