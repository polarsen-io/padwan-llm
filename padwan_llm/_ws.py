from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

import niquests

from ._json import dumps as _json_dumps, loads as _json_loads
from .logs import log

__all__ = ("READ_POLL_INTERVAL", "WsConnection", "enable_read_polling")

# The transport locks the socket per task, so a parked read starves senders.
# A socket read timeout makes the read release the lock periodically; the
# transport treats it as clean ("ws algorithms based on timeouts") and the
# connection iterator swallows the tick. Bounds send latency on a quiet socket;
# roughly one queued send gets through per tick, so senders should batch.
READ_POLL_INTERVAL = 0.1


def enable_read_polling(ext: Any, interval: float) -> None:
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


@dataclass
class WsConnection:
    """JSON-over-WebSocket connection: send events, async-iterate parsed messages.

    The iterator swallows the periodic read-poll ticks armed by
    :func:`enable_read_polling`, each of which releases the socket so queued
    sends go out within :data:`READ_POLL_INTERVAL` seconds.
    """

    ext: Any
    _closed: bool = False

    async def send_event(self, event: Mapping[str, Any]) -> None:
        """Send a raw client event as a JSON text frame.

        Safe to call while another task awaits server events: the frame goes out
        at the next read-poll tick, within :data:`READ_POLL_INTERVAL` seconds.
        """
        await self.ext.send_payload(_json_dumps(event))

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
