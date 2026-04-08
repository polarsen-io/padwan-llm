import asyncio
import enum
import functools
import json
import os.path
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypedDict,
    cast,
    runtime_checkable,
)
from http import HTTPStatus
from urllib.parse import urlparse

from importlib.metadata import version as _pkg_version

import niquests

from ._json import loads as _json_loads
from .logs import log
from .models import ToolDefinition

__version__ = _pkg_version("padwan-llm")

__all__ = ("McpTool", "McpTransport", "McpStreamable", "McpStdio", "ProgressEvent")

_JSONRPC = "2.0"
_PROTOCOL_VERSION = "2025-11-25"
_MAX_LISTEN_RETRIES = 5
_DEFAULT_RETRY_MS = 3000


class _Notification(enum.StrEnum):
    TOOLS_CHANGED = "notifications/tools/list_changed"
    PROGRESS = "notifications/progress"
    INITIALIZED = "notifications/initialized"
    CANCELLED = "notifications/cancelled"


type _RpcMethod = Literal["initialize", "tools/list", "tools/call", "ping"]


class ProgressEvent(TypedDict):
    """Progress notification from an MCP server."""

    progressToken: str | int
    progress: float
    total: NotRequired[float]
    message: NotRequired[str]


class _JsonRpcNotification(TypedDict):
    jsonrpc: str
    method: str
    params: NotRequired[dict[str, Any]]


class _JsonRpcRequest(_JsonRpcNotification):
    id: str


def _to_sse_url(url: str) -> str:
    """Convert http(s):// URL to the niquests SSE scheme (psse:// or sse://)."""
    stripped = url.removeprefix("https://")
    if stripped is not url:
        return "sse://" + stripped
    return "psse://" + url.removeprefix("http://")


@dataclass
class McpTool:
    """A single MCP-compatible tool with name, schema, and async handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    """JSON Schema describing the tool's expected parameters. Corresponds to
    the ``inputSchema`` field in the MCP wire protocol (camelCase on the wire,
    snake_case here)."""
    handler: Callable[[dict[str, Any]], Any] = field(
        repr=False, default=lambda args: args
    )
    """Async callable invoked when the LLM requests this tool. Receives the
    parsed arguments dict and should return the tool result. Defaults to an
    identity function (pass-through), useful when only the schema matters."""

    def to_tool_def(self) -> ToolDefinition:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }


@runtime_checkable
class McpTransport(Protocol):
    """Structural interface satisfied by `McpStreamable` and `McpStdio`.

    An MCP transport exposes a `tools` property (a list refreshed in place
    on `notifications/tools/list_changed`), an `is_open` property so callers
    can detect an already-entered instance, and is usable as an async
    context manager so its session can be initialized on entry and torn
    down on exit.

    `name_prefix` lets the caller pin a stable, human-readable namespace
    for tools coming from this transport (e.g. ``"weather"`` produces
    ``weather__forecast``). When ``None``, `auto_prefix` exposes a
    fallback derived from the transport's identity (URL host for
    streamable, ``args[0]``/command basename for stdio) — `AgentSession`
    only applies it when an actual collision is detected, so
    single-transport setups keep clean bare names.
    """

    name_prefix: str | None

    @property
    def tools(self) -> list[McpTool]: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def auto_prefix(self) -> str: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args: object) -> None: ...


# Shared helpers


def _sanitize_prefix(s: str) -> str:
    """Make a string safe for use as a function-name prefix.

    Provider function-name regexes (OpenAI, Gemini, Anthropic) only allow
    `[a-zA-Z0-9_-]`. Replace anything else with underscore, collapse runs,
    and never return empty (fall back to ``"mcp"``).
    """
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", s).strip("_")
    return cleaned or "mcp"


def _build_tools(
    raw_tools: list[dict[str, Any]],
    call_fn: Callable,
    name_prefix: str | None = None,
) -> list[McpTool]:
    """Build McpTool list from a tools/list result.

    When `name_prefix` is not ``None``, every tool's public name becomes
    ``f"{name_prefix}__{wire_name}"`` while the handler stays bound to
    the underlying wire name. Lets a caller expose a transport's tools
    under a stable namespace without confusing the dispatch path.
    """
    return [
        McpTool(
            name=(
                f"{name_prefix}__{t['name']}" if name_prefix is not None else t["name"]
            ),
            description=t.get("description") or "",
            input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
            handler=lambda args, _n=t["name"]: call_fn(_n, args),
        )
        for t in raw_tools
    ]


def _normalize_call_result(result: Any) -> dict[str, Any]:
    """Pass a `tools/call` result through unchanged, only normalising the
    `content` field's presence.

    The MCP `tools/call` response can carry far more than just `text`
    content blocks: `image`/`resource` blocks, an `isError` flag, and
    `structuredContent` for machine-readable tools (the whole point of
    JSON-only MCP servers). The previous implementation rebuilt the
    result from text blocks only, silently dropping every other field
    and turning a JSON-only tool into `{"content": []}` before the
    agent ever saw it. Now we return the dict as-is so downstream
    consumers (`agent._extract_text`) can decide how to reduce it.
    """
    if not isinstance(result, dict):
        return {"content": []}
    return result


def _mcp_headers(
    accept: str = "application/json",
    session_id: str | None = None,
    token: str | None = None,
) -> dict[str, str]:
    """Build MCP HTTP headers with optional session ID and bearer token."""
    h = {
        "Content-Type": "application/json",
        "Accept": accept,
        "MCP-Protocol-Version": _PROTOCOL_VERSION,
    }
    if session_id:
        h["MCP-Session-Id"] = session_id
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _check_rpc_error(data: dict[str, Any]) -> None:
    """Raise if a JSON-RPC response contains an error."""
    if err := data.get("error"):
        raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")


def _check_protocol_version(init_result: dict[str, Any] | None) -> None:
    """Warn if the server's negotiated protocol version differs from ours.

    Per spec, if the server responds with a different version, the client
    SHOULD disconnect if it cannot support it. We log a warning instead of
    hard-failing to stay permissive across minor spec revisions.
    """
    if not init_result:
        return
    server_version = init_result.get("protocolVersion")
    if server_version and server_version != _PROTOCOL_VERSION:
        log.warning(
            "MCP protocol version mismatch: client=%s server=%s",
            _PROTOCOL_VERSION,
            server_version,
        )


# McpStreamable — streamable-HTTP transport


@dataclass
class McpStreamable:
    """MCP client over streamable-HTTP using niquests.

    POST JSON-RPC for all requests; response is JSON or SSE stream.
    Background GET SSE stream for server-push notifications.

    Spec: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#streamable-http
    """

    url: str
    token: str | None = None
    on_progress: Callable[[ProgressEvent], Any] | None = None
    client_name: str = "padwan-llm"
    client_version: str = __version__
    name_prefix: str | None = None
    """Optional namespace for this transport's tools. When set, every
    discovered tool is renamed to ``f"{name_prefix}__{wire_name}"`` so
    multiple transports can coexist without colliding. Leave as ``None``
    to use bare wire names; `AgentSession` will fall back to `auto_prefix`
    if (and only if) it detects a collision at runtime."""
    _tools: list[McpTool] = field(init=False, default_factory=list)
    _http: niquests.AsyncSession = field(
        init=False, default_factory=niquests.AsyncSession
    )
    _session_id: str | None = field(init=False, default=None)
    _bg_task: asyncio.Task[None] | None = field(init=False, default=None)
    _last_event_id: str | None = field(init=False, default=None)
    _retry_ms: int = field(init=False, default=_DEFAULT_RETRY_MS)

    @functools.cached_property
    def _sse_url(self) -> str:
        return _to_sse_url(self.url)

    @property
    def is_open(self) -> bool:
        """True once the transport has been entered and the listener task is live."""
        return self._bg_task is not None

    @property
    def auto_prefix(self) -> str:
        """Fallback prefix derived from the URL host (e.g.
        ``mcp.data.gouv.fr`` → ``mcp_data_gouv_fr``). Only used by
        `AgentSession` when a collision forces disambiguation."""
        host = urlparse(self.url).hostname or "mcp"
        return _sanitize_prefix(host)

    async def __aenter__(self) -> Self:
        if self._bg_task is not None:
            raise RuntimeError(
                "McpStreamable is already open; re-entering is not supported"
            )
        try:
            await self._initialize()
            await self._refresh_tools()
            self._bg_task = asyncio.create_task(self._listen())
        except BaseException:
            # If initialize succeeded but refresh_tools failed (or the
            # listener failed to schedule), the server has already handed
            # us a session id and our HTTP client is still open. `__aexit__`
            # would never run because this method is raising, so clean up
            # everything ourselves before propagating. Swallow cleanup
            # failures so they don't mask the original error.
            try:
                await self.__aexit__(None, None, None)
            except Exception:
                log.exception("MCP cleanup after failed startup raised")
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        try:
            if self._bg_task:
                self._bg_task.cancel()
                try:
                    await self._bg_task
                except asyncio.CancelledError:
                    log.debug("MCP GET listener cancelled")
            if self._session_id:
                await self._http.delete(self.url, headers=self._headers())
        except Exception:
            log.warning("MCP session cleanup failed", exc_info=True)
            raise
        finally:
            await self._http.close()
            # Reset state so the same instance can be re-entered in a
            # fresh `async with` block. `is_open` must flip back to False
            # so callers (e.g. AgentSession.__aenter__) treat this as a
            # brand-new transport on the next entry.
            self._bg_task = None
            self._session_id = None
            self._last_event_id = None
            self._retry_ms = _DEFAULT_RETRY_MS
            self._http = niquests.AsyncSession()

    @property
    def tools(self) -> list[McpTool]:
        return self._tools

    def _headers(self, accept: str = "application/json") -> dict[str, str]:
        return _mcp_headers(accept, self._session_id, self.token)

    def _dispatch_progress(self, params: dict[str, Any]) -> None:
        if self.on_progress is not None:
            self.on_progress(params)  # type: ignore[arg-type]

    async def _rpc(
        self,
        method: _RpcMethod,
        params: dict[str, Any] | None = None,
        *,
        _reinit: bool = True,
    ) -> Any:
        """POST a JSON-RPC request and return the result.

        Handles both JSON and SSE response formats. On 404 with an active
        session, re-initializes once and retries (per spec).
        """
        req_id = uuid.uuid4().hex
        payload: _JsonRpcRequest = {"jsonrpc": _JSONRPC, "id": req_id, "method": method}
        if params is not None:
            payload["params"] = params
        r = await self._http.post(
            self._sse_url,
            json=payload,
            headers=self._headers(accept="application/json, text/event-stream"),
        )
        if r.status_code == HTTPStatus.UNAUTHORIZED:
            raise RuntimeError("MCP server requires authorization (HTTP 401)")
        # Server lost track of session
        if r.status_code == HTTPStatus.NOT_FOUND and self._session_id and _reinit:
            self._session_id = None
            await self._initialize()
            return await self._rpc(method, params, _reinit=False)
        r.raise_for_status()
        if sid := r.headers.get("MCP-Session-Id"):
            self._session_id = cast(str, sid)
        ct = cast(str, r.headers.get("content-type", ""))
        if "text/event-stream" in ct:
            return await self._read_sse_response(r)
        data: dict[str, Any] = await r.json()
        _check_rpc_error(data)
        return data.get("result")

    async def _read_sse_response(self, r: niquests.Response) -> Any:
        """Consume SSE stream from a POST response, return the JSON-RPC result.

        The spec allows servers to send notifications/requests before the
        response. We process all events and return the result from the first
        JSON-RPC response (message with 'result' or 'error').
        """
        ext = r.extension
        if ext is None:
            raise RuntimeError("SSE extension not available on response")
        while not ext.closed:
            event = await ext.next_payload()
            if event is None:
                break
            if event.id:
                self._last_event_id = event.id
            if event.retry is not None:
                self._retry_ms = event.retry
            # Empty `data:` frames are valid SSE keep-alives, not malformed
            # JSON. Skip them before attempting to parse.
            if not event.data:
                continue
            try:
                data: dict[str, Any] = event.json()
            except ValueError:
                log.warning("MCP: ignoring malformed SSE event: %s", event.data)
                continue
            if "result" in data or "error" in data:
                _check_rpc_error(data)
                return data.get("result")
            method = data.get("method")
            if method == _Notification.TOOLS_CHANGED:
                await self._refresh_tools()
            elif method == _Notification.PROGRESS and data.get("params"):
                self._dispatch_progress(data["params"])
        raise RuntimeError("SSE stream ended without a JSON-RPC response")

    async def _listen(self, max_retries: int = _MAX_LISTEN_RETRIES) -> None:
        """GET SSE stream for server-push notifications, with reconnection.

        Spec: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#listening-for-messages-from-the-server
        """
        retries = 0
        while retries < max_retries:
            try:
                headers = self._headers(accept="text/event-stream")
                if self._last_event_id is not None:
                    headers["Last-Event-ID"] = self._last_event_id
                r = await self._http.get(self._sse_url, headers=headers)
                if r.status_code == HTTPStatus.METHOD_NOT_ALLOWED:
                    return
                ext = (r.raise_for_status()).extension
                if ext is None:
                    return
                retries = 0
                while not ext.closed:
                    event = await ext.next_payload()
                    if event is None:
                        break
                    if event.id:
                        self._last_event_id = event.id
                    if event.retry is not None:
                        self._retry_ms = event.retry
                    # Empty `data:` frames are valid SSE keep-alives, not
                    # malformed JSON. Skip them silently — servers (e.g.
                    # https://mcp.data.gouv.fr/mcp) emit them every few
                    # seconds to keep the connection alive, and parsing
                    # them would flood the logs with bogus warnings.
                    if not event.data:
                        continue
                    try:
                        msg: dict[str, Any] = event.json()
                    except ValueError:
                        log.warning("MCP: ignoring malformed SSE event: %s", event.data)
                        continue
                    method = msg.get("method")
                    if method == _Notification.TOOLS_CHANGED:
                        await self._refresh_tools()
                    elif method == _Notification.PROGRESS and msg.get("params"):
                        self._dispatch_progress(msg["params"])
                retries += 1
                log.debug(
                    "MCP GET stream ended, reconnecting (%d/%d)",
                    retries,
                    max_retries,
                )
                await asyncio.sleep(self._retry_ms / 1000)
            except asyncio.CancelledError:
                log.debug("MCP GET listener cancelled")
                return
            except Exception:
                retries += 1
                log.warning(
                    "MCP GET listener error (%d/%d)",
                    retries,
                    max_retries,
                    exc_info=True,
                )
                await asyncio.sleep(self._retry_ms / 1000)

    async def _initialize(self) -> None:
        """Perform the MCP initialization handshake.

        See https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle#initialization
        """
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
            _reinit=False,
        )
        _check_protocol_version(result)
        await self._http.post(
            self.url,
            json=_JsonRpcNotification(
                jsonrpc=_JSONRPC, method=_Notification.INITIALIZED
            ),
            headers=self._headers(),
        )

    async def _refresh_tools(self) -> None:
        # Mutate the list in place so external references (e.g. AgentSession
        # holding onto `mcp.tools`) see the refreshed contents.
        result = await self._rpc("tools/list")
        self._tools[:] = _build_tools(
            result.get("tools", []), self._call, self.name_prefix
        )

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        return _normalize_call_result(result)

    async def ping(self) -> None:
        """Send a ping request to verify the server is responsive."""
        await self._rpc("ping")

    async def cancel(self, request_id: str, reason: str | None = None) -> None:
        """Send notifications/cancelled to cancel an in-flight request."""
        params: dict[str, Any] = {"requestId": request_id}
        if reason is not None:
            params["reason"] = reason
        await self._http.post(
            self.url,
            json=_JsonRpcNotification(
                jsonrpc=_JSONRPC,
                method=_Notification.CANCELLED,
                params=params,
            ),
            headers=self._headers(),
        )


# McpStdio — stdio transport


@dataclass
class McpStdio:
    """MCP client over stdio, spawning the server as a subprocess.

    Communicates via newline-delimited JSON-RPC on stdin/stdout.

    Spec: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports#stdio
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    on_progress: Callable[[ProgressEvent], Any] | None = None
    client_name: str = "padwan-llm"
    client_version: str = __version__
    name_prefix: str | None = None
    """Optional namespace for this transport's tools (see
    `McpStreamable.name_prefix`)."""

    _tools: list[McpTool] = field(init=False, default_factory=list)
    _process: asyncio.subprocess.Process | None = field(init=False, default=None)
    _next_id: int = field(init=False, default=0)
    _pending: dict[str, asyncio.Future[Any]] = field(init=False, default_factory=dict)
    _reader_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stderr_task: asyncio.Task[None] | None = field(init=False, default=None)
    _loop: asyncio.AbstractEventLoop = field(init=False)

    @property
    def auto_prefix(self) -> str:
        """Fallback prefix derived from `args[0]` if present, otherwise
        from the command basename. The first arg is usually the actual
        server identity (script path, npm package name) — much more
        readable than ``"npx"`` or ``"python"``. Stripped of common
        suffixes (`.py`, `.js`) and namespace prefixes
        (e.g. ``@modelcontextprotocol/``)."""
        source = self.args[0] if self.args else self.command
        base = os.path.basename(source)
        # Drop common script extensions and npm scope prefixes so the
        # prefix doesn't carry packaging noise into prompts.
        for suffix in (".py", ".js", ".ts", ".mjs"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base.startswith("@") and "/" in base:
            base = base.split("/", 1)[1]
        return _sanitize_prefix(base)

    @property
    def is_open(self) -> bool:
        """True while the child process is running and the reader task is live."""
        return self._process is not None

    async def __aenter__(self) -> Self:
        if self._process is not None:
            raise RuntimeError("McpStdio is already open; re-entering is not supported")
        self._loop = asyncio.get_running_loop()
        try:
            self._process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self.env,
                cwd=self.cwd,
            )
            self._reader_task = asyncio.create_task(self._reader())
            self._stderr_task = asyncio.create_task(self._drain_stderr())
            await self._initialize()
            await self._refresh_tools()
        except BaseException:
            # A failure after spawning the subprocess (bad handshake,
            # crash, malformed JSON-RPC) would otherwise leak the child
            # and both background tasks — `__aexit__` never runs because
            # this method is raising, so clean up before propagating.
            try:
                await self.__aexit__(None, None, None)
            except Exception:
                log.exception("MCP stdio cleanup after failed startup raised")
            raise
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._stderr_task:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        if self._process:
            self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._process.kill()
                    # SIGKILL is reliable but POSIX still requires us to
                    # call wait() to reap the zombie. Without this, every
                    # wedged child becomes a defunct process the parent
                    # never collects, and callers that recycle stdio
                    # transports accumulate them. Bounded so a truly
                    # un-killable child can't hang shutdown.
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=2.0)
                    except asyncio.TimeoutError:
                        log.warning("MCP stdio child did not exit after SIGKILL")
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("MCP stdio connection closed"))
        self._pending.clear()
        # Reset state so the same instance can be re-entered in a fresh
        # `async with` block. `is_open` must flip back to False so callers
        # (e.g. AgentSession.__aenter__) treat this as a brand-new
        # transport on the next entry.
        self._process = None
        self._reader_task = None
        self._stderr_task = None
        self._next_id = 0

    @property
    def tools(self) -> list[McpTool]:
        return self._tools

    async def _send(self, msg: _JsonRpcNotification | _JsonRpcRequest) -> None:
        if not self._process:
            raise RuntimeError("MCP stdio process not running")
        # Newline-Delimited JSON framing
        data = json.dumps(msg).encode() + b"\n"
        self._process.stdin.write(data)
        await self._process.stdin.drain()

    async def _drain_stderr(self) -> None:
        """Background task that forwards the child process's stderr to logs.

        Stderr is piped so the parent can observe it, but if nothing reads
        from the pipe the buffer fills (typically ~64 KB) and the child
        blocks on its next stderr write — wedging the whole session.
        """
        try:
            async for raw_line in self._process.stderr:
                line = raw_line.decode(errors="replace").rstrip()
                if line:
                    log.debug("MCP stdio stderr: %s", line)
        except asyncio.CancelledError:
            log.debug("MCP stdio stderr drain cancelled")
        except Exception:
            log.warning("MCP stdio stderr drain failed", exc_info=True)

    async def _rpc(
        self, method: _RpcMethod, params: dict[str, Any] | None = None
    ) -> Any:
        self._next_id += 1
        rid = str(self._next_id)
        payload: _JsonRpcRequest = {"jsonrpc": _JSONRPC, "id": rid, "method": method}
        if params is not None:
            payload["params"] = params
        fut: asyncio.Future[Any] = self._loop.create_future()
        self._pending[rid] = fut
        try:
            await self._send(payload)
            return await fut
        finally:
            # Drop the entry on cancel/exception so a late response from the
            # server doesn't try to resolve a stale (possibly cancelled) future.
            self._pending.pop(rid, None)

    async def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: _JsonRpcNotification = {"jsonrpc": _JSONRPC, "method": method}
        if params is not None:
            payload["params"] = params
        await self._send(payload)

    async def _reader(self) -> None:
        """Read stdout line-by-line and dispatch JSON-RPC messages."""
        try:
            async for raw_line in self._process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg: dict[str, Any] = _json_loads(line)
                except json.JSONDecodeError:
                    log.warning("MCP stdio: ignoring malformed line: %s", line)
                    continue
                msg_id = msg.get("id")
                if msg_id is not None and str(msg_id) in self._pending:
                    fut = self._pending.pop(str(msg_id))
                    if "error" in msg:
                        err = msg["error"]
                        fut.set_exception(
                            RuntimeError(
                                f"MCP error {err.get('code')}: {err.get('message')}"
                            )
                        )
                    else:
                        fut.set_result(msg.get("result"))
                else:
                    method = msg.get("method")
                    if method == _Notification.TOOLS_CHANGED:
                        asyncio.create_task(self._refresh_tools())
                    elif (
                        method == _Notification.PROGRESS
                        and msg.get("params")
                        and self.on_progress is not None
                    ):
                        self.on_progress(msg["params"])  # type: ignore[arg-type]
        except asyncio.CancelledError:
            log.debug("MCP stdio reader cancelled")
        except Exception:
            log.error("MCP stdio reader failed", exc_info=True)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(RuntimeError("MCP stdio reader stopped"))

    async def _initialize(self) -> None:
        """Perform the MCP initialization handshake.

        See https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle#initialization
        """
        result = await self._rpc(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )
        _check_protocol_version(result)
        await self._notify(_Notification.INITIALIZED)

    async def _refresh_tools(self) -> None:
        # Mutate the list in place so external references (e.g. AgentSession
        # holding onto `mcp.tools`) see the refreshed contents.
        result = await self._rpc("tools/list")
        self._tools[:] = _build_tools(
            result.get("tools", []), self._call, self.name_prefix
        )

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        return _normalize_call_result(result)

    async def ping(self) -> None:
        """Send a ping request to verify the server is responsive."""
        await self._rpc("ping")

    async def cancel(self, request_id: str, reason: str | None = None) -> None:
        """Send notifications/cancelled to cancel an in-flight request."""
        params: dict[str, Any] = {"requestId": request_id}
        if reason is not None:
            params["reason"] = reason
        await self._notify(_Notification.CANCELLED, params)
