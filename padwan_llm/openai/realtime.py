from __future__ import annotations

import base64
import enum
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args

import niquests

from .._base import LLMError, RealtimeClientBase, env_api_key
from .._json import dumps as _json_dumps, loads as _json_loads
from ..errors import Provider
from ..logs import log

__all__ = (
    "REALTIME_ENDPOINT",
    "DEFAULT_REALTIME_MODEL",
    "REALTIME_SAMPLE_RATE",
    "NO_TURN_DETECTION",
    "RealtimeVoice",
    "RealtimeServerEvent",
    "OpenAIRealtimeClient",
    "RealtimeConnection",
)

REALTIME_ENDPOINT = "wss://api.openai.com/v1/realtime"
DEFAULT_REALTIME_MODEL = "gpt-realtime"

# gpt-realtime exchanges mono little-endian PCM16 at 24 kHz.
REALTIME_SAMPLE_RATE = 24_000

# Pass as ``turn_detection`` to disable server VAD (manual / push-to-talk mode);
# you then drive each turn yourself with commit_audio() + create_response().
NO_TURN_DETECTION = "none"

# The transport locks the socket per task, so a parked read starves senders.
# A socket read timeout makes the read release the lock periodically; the
# transport treats it as clean ("ws algorithms based on timeouts") and the
# connection iterator swallows the tick. Bounds send latency on a quiet socket;
# roughly one queued send gets through per tick, so senders should batch.
_READ_POLL_INTERVAL = 0.1


def _enable_read_polling(ext: Any, interval: float) -> None:
    """Arm the ws socket read timeout that drives send/receive interleaving.

    Reaches through transport internals (the connection behind the extension's
    stream reader) because no public knob exists (see
    https://github.com/jawah/urllib3.future/issues/400); best effort — without
    it, sends block until the server happens to emit an event.
    """
    try:
        conn = ext._dsa._read.__self__
        conn.timeout = interval
        conn.sock.settimeout(interval)
    except AttributeError:
        log.debug("could not arm ws read polling", exc_info=True)


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
class OpenAIRealtimeClient(RealtimeClientBase):
    """Speech-to-speech client for the OpenAI Realtime API (``gpt-realtime``) over a WebSocket.

    Requires the ``realtime`` extra. *api_key* falls back to ``OPENAI_API_KEY``;
    *timeout* bounds only the opening handshake — the open socket tolerates a
    silent model between turns. Wire shapes:
    https://platform.openai.com/docs/guides/realtime
    """

    provider: ClassVar[Provider] = "openai"

    model: str = DEFAULT_REALTIME_MODEL
    base_url: str = REALTIME_ENDPOINT

    def _get_default_api_key(self) -> str:
        return env_api_key(self.provider, "OPENAI_API_KEY")

    def _set_auth_headers(self, session: niquests.AsyncSession) -> None:
        session.headers["Authorization"] = f"Bearer {self._api_key}"

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

        The client must be open (``async with client:``); each call opens an
        independent connection over the client's session.
        *turn_detection* defaults to server-side VAD; pass a mapping to override,
        or :data:`NO_TURN_DETECTION` to drive turns manually (push-to-talk) with
        :meth:`RealtimeConnection.commit_audio` + :meth:`RealtimeConnection.create_response`.
        *transcription_model* ``None`` disables transcription of your own speech.
        """
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
        _enable_read_polling(ext, _READ_POLL_INTERVAL)
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

    The transport allows only one task on the socket at a time, so a read parked
    in ``next_payload`` would block every send until the server happens to emit
    an event. ``connect()`` therefore arms a short socket read timeout: the
    iterator silently swallows the periodic timeout ticks, and each tick
    releases the socket so queued sends go out within ``_READ_POLL_INTERVAL``.
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
        """Send a raw client event as a JSON text frame.

        Safe to call while another task awaits server events: the frame goes out
        at the next read-poll tick, within ``_READ_POLL_INTERVAL`` seconds.
        """
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
            try:
                payload = await self.ext.next_payload()
            except niquests.exceptions.ReadTimeout:
                continue  # read-poll tick: lets queued sends borrow the socket
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
