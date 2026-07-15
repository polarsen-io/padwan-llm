from __future__ import annotations

import base64
import enum
import os
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import niquests

from .._base import LLMError
from .._json import dumps as _json_dumps, loads as _json_loads
from ..logs import log

__all__ = (
    "REALTIME_ENDPOINT",
    "DEFAULT_REALTIME_MODEL",
    "REALTIME_SAMPLE_RATE",
    "NO_TURN_DETECTION",
    "RealtimeVoice",
    "RealtimeServerEvent",
    "RealtimeClient",
    "RealtimeConnection",
)

REALTIME_ENDPOINT = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_MODEL = "gpt-realtime"

# gpt-realtime exchanges mono little-endian PCM16 at 24 kHz.
REALTIME_SAMPLE_RATE = 24_000

# Pass as ``turn_detection`` to disable server VAD (manual / push-to-talk mode);
# you then drive each turn yourself with commit_audio() + create_response().
NO_TURN_DETECTION = "none"

# https://platform.openai.com/docs/guides/realtime — "marin" and "cedar" are the
# voices introduced with gpt-realtime; the rest carry over from the preview models.
type RealtimeVoice = str
_VOICES = (
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
)


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
class RealtimeClient:
    """Speech-to-speech client for the OpenAI Realtime API over a WebSocket.

    Talks to the GA ``gpt-realtime`` model. The transport is niquests' native
    WebSocket support (requires the ``ws`` extra, i.e. ``padwan-llm[realtime]``).
    Pass *api_key* explicitly or leave it unset to read ``OPENAI_API_KEY`` from the
    environment. *timeout* bounds the opening handshake; reads on the open socket
    are left unbounded so the model can stay silent between turns without the
    connection being torn down.
    """

    model: str = DEFAULT_REALTIME_MODEL
    api_key: str | None = None
    base_url: str = REALTIME_ENDPOINT
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self._api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise LLMError("openai", "OPENAI_API_KEY not set")

    @asynccontextmanager
    async def connect(
        self,
        *,
        instructions: str | None = None,
        voice: RealtimeVoice = "marin",
        turn_detection: Mapping[str, Any] | str | None = None,
        transcription_model: str | None = "whisper-1",
        output_modalities: Sequence[str] = ("audio",),
        sample_rate: int = REALTIME_SAMPLE_RATE,
    ) -> AsyncIterator[RealtimeConnection]:
        """Open a configured realtime session and yield a :class:`RealtimeConnection`.

        *instructions* is the system prompt steering the voice agent and *voice*
        selects the spoken voice. *turn_detection* defaults to server-side VAD so
        the model decides when you have stopped talking; pass an explicit mapping
        (e.g. ``{"type": "semantic_vad"}``) to override, or :data:`NO_TURN_DETECTION`
        to disable VAD and drive each turn manually (push-to-talk) with
        :meth:`RealtimeConnection.commit_audio` + :meth:`RealtimeConnection.create_response`.
        *transcription_model* enables transcription of your own speech (set to
        ``None`` to disable). The session is configured before control returns and
        closed automatically on exit.
        """
        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with niquests.AsyncSession() as session:
            # Bounds only the upgrade handshake; the ws extension reads frames with
            # recv_extended(None), so idle gaps between turns never abort next_payload.
            resp = await session.get(url, headers=headers, timeout=self.timeout)
            ext = resp.extension
            if ext is None:
                raise LLMError(
                    "openai",
                    f"realtime handshake did not upgrade to a websocket "
                    f"(status {resp.status_code})",
                )
            conn = RealtimeConnection(ext, sample_rate=sample_rate)
            await conn.configure(
                instructions=instructions,
                voice=voice,
                turn_detection=turn_detection,
                transcription_model=transcription_model,
                output_modalities=list(output_modalities),
            )
            try:
                yield conn
            finally:
                await conn.close()


@dataclass
class RealtimeConnection:
    """An open realtime session.

    Stream microphone audio with :meth:`append_audio` and consume server events by
    async-iterating the connection. Audio is mono little-endian PCM16 at
    *sample_rate* Hz in both directions.
    """

    ext: Any
    sample_rate: int = REALTIME_SAMPLE_RATE
    _closed: bool = False

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
        unit-tested without a live connection. *turn_detection* of
        :data:`NO_TURN_DETECTION` emits a JSON ``null`` to disable server VAD; a
        falsy value uses the ``server_vad`` default; a mapping is sent verbatim.
        """
        audio_format = {"type": "audio/pcm", "rate": self.sample_rate}
        if isinstance(turn_detection, str):
            detection: dict[str, Any] | None = None  # NO_TURN_DETECTION -> JSON null
        elif turn_detection:
            detection = dict(turn_detection)
        else:
            detection = {"type": "server_vad"}
        audio_input: dict[str, Any] = {
            "format": audio_format,
            "turn_detection": detection,
        }
        if transcription_model:
            audio_input["transcription"] = {"model": transcription_model}
        session: dict[str, Any] = {
            "type": "realtime",
            "output_modalities": list(output_modalities),
            "audio": {
                "input": audio_input,
                "output": {"format": audio_format, "voice": voice},
            },
        }
        if instructions:
            session["instructions"] = instructions
        return {"type": "session.update", "session": session}

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

    async def send_event(self, event: Mapping[str, Any]) -> None:
        """Send a raw client event as a JSON text frame."""
        await self.ext.send_payload(_json_dumps(event))

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

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        while not self._closed:
            payload = await self.ext.next_payload()
            if payload is None:
                break
            if isinstance(payload, (bytes, bytearray)):
                payload = bytes(payload).decode("utf-8")
            yield _json_loads(payload)

    async def close(self) -> None:
        """Close the underlying websocket (idempotent)."""
        if self._closed:
            return
        self._closed = True
        try:
            await self.ext.close()
        except Exception:  # best effort — the socket may already be gone
            log.debug("error closing realtime websocket", exc_info=True)
