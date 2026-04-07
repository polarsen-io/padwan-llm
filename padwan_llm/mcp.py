import asyncio
import enum
import functools
import json
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

from importlib.metadata import version as _pkg_version

import niquests

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
    on `notifications/tools/list_changed`) and is usable as an async context
    manager so its session can be initialized on entry and torn down on exit.
    """

    @property
    def tools(self) -> list[McpTool]: ...
    async def __aenter__(self) -> Self: ...
    async def __aexit__(self, *args: object) -> None: ...


# Shared helpers


def _build_tools(raw_tools: list[dict[str, Any]], call_fn: Callable) -> list[McpTool]:
    """Build McpTool list from a tools/list result."""
    return [
        McpTool(
            name=t["name"],
            description=t.get("description") or "",
            input_schema=t.get("inputSchema") or {"type": "object", "properties": {}},
            handler=lambda args, _n=t["name"]: call_fn(_n, args),
        )
        for t in raw_tools
    ]


def _extract_text_content(result: dict[str, Any]) -> dict[str, Any]:
    """Extract text content blocks from a tools/call result."""
    return {
        "content": [
            {"type": c["type"], "text": c.get("text", "")}
            for c in result.get("content", [])
            if c.get("type") == "text"
        ]
    }


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

    async def __aenter__(self) -> Self:
        await self._initialize()
        await self._refresh_tools()
        self._bg_task = asyncio.create_task(self._listen())
        return self

    async def __aexit__(self, *_: object) -> None:
        try:
            # Cleaning Background Task
            if self._bg_task:
                self._bg_task.cancel()
                try:
                    await self._bg_task
                except asyncio.CancelledError:
                    log.debug("MCP GET listener cancelled")
            #
            if self._session_id:
                await self._http.delete(self.url, headers=self._headers())
        except Exception:
            log.warning("MCP session cleanup failed", exc_info=True)
            raise
        finally:
            await self._http.close()

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
            data: dict[str, Any] = event.json()
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
        self._tools[:] = _build_tools(result.get("tools", []), self._call)

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        return _extract_text_content(result)

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

    _tools: list[McpTool] = field(init=False, default_factory=list)
    _process: asyncio.subprocess.Process | None = field(init=False, default=None)
    _next_id: int = field(init=False, default=0)
    _pending: dict[str, asyncio.Future[Any]] = field(init=False, default_factory=dict)
    _reader_task: asyncio.Task[None] | None = field(init=False, default=None)
    _stderr_task: asyncio.Task[None] | None = field(init=False, default=None)
    _loop: asyncio.AbstractEventLoop = field(init=False)

    async def __aenter__(self) -> Self:
        self._loop = asyncio.get_running_loop()
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
            if self._process.stdin:
                self._process.stdin.close()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._process.kill()
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("MCP stdio connection closed"))
        self._pending.clear()

    @property
    def tools(self) -> list[McpTool]:
        return self._tools

    async def _send(self, msg: _JsonRpcNotification | _JsonRpcRequest) -> None:
        if not self._process or not self._process.stdin:
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
        if not self._process or not self._process.stderr:
            return
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
        if not self._process or not self._process.stdout:
            return
        try:
            async for raw_line in self._process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("MCP stdio: ignoring malformed line: %s", line)
                    continue
                msg_id = msg.get("id")
                if msg_id is not None and str(msg_id) in self._pending:
                    fut = self._pending.pop(str(msg_id))
                    if fut.done():
                        log.debug(
                            "MCP stdio: dropping late response for id=%s "
                            "(caller already cancelled or resolved)",
                            msg_id,
                        )
                        continue
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
        self._tools[:] = _build_tools(result.get("tools", []), self._call)

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self._rpc("tools/call", {"name": name, "arguments": args})
        return _extract_text_content(result)

    async def ping(self) -> None:
        """Send a ping request to verify the server is responsive."""
        await self._rpc("ping")

    async def cancel(self, request_id: str, reason: str | None = None) -> None:
        """Send notifications/cancelled to cancel an in-flight request."""
        params: dict[str, Any] = {"requestId": request_id}
        if reason is not None:
            params["reason"] = reason
        await self._notify(_Notification.CANCELLED, params)
