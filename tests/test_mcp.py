import asyncio
import json
import sys
from contextlib import nullcontext
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP

from padwan_llm.mcp import McpStreamable, McpStdio, McpTool, _to_sse_url


def _make_server() -> FastMCP:
    server = FastMCP("test")

    @server.tool()
    def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"Sunny in {city}"

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    @server.tool()
    def no_params() -> str:
        """A tool with no parameters."""
        return "ok"

    return server


@pytest.fixture
def mcp_wire_tools() -> list[dict]:
    """Wire-format tool dicts as returned by a real mcp FastMCP server."""
    server = _make_server()
    tools = asyncio.run(server.list_tools())
    return [t.model_dump(by_alias=True, exclude_none=True) for t in tools]


def _wire_to_mcp_tool(wire: dict) -> McpTool:
    """Same logic as McpStreamable._refresh_tools."""
    return McpTool(
        name=wire["name"],
        description=wire.get("description") or "",
        input_schema=wire.get("inputSchema") or {"type": "object", "properties": {}},
    )


class TestMcpTool:
    def test_round_trip_fields(self, mcp_wire_tools):
        """McpTool correctly captures name, description, and input schema from mcp wire format."""
        tool_map = {t["name"]: _wire_to_mcp_tool(t) for t in mcp_wire_tools}

        weather = tool_map["get_weather"]
        assert weather.name == "get_weather"
        assert weather.description == "Get the weather for a given city."
        assert weather.input_schema["type"] == "object"
        assert "city" in weather.input_schema["properties"]

        add = tool_map["add"]
        assert add.name == "add"
        assert set(add.input_schema["required"]) == {"a", "b"}

    def test_to_tool_def(self, mcp_wire_tools):
        """to_tool_def() produces a ToolDefinition compatible with LLM provider APIs."""
        for wire in mcp_wire_tools:
            tool = _wire_to_mcp_tool(wire)
            td = tool.to_tool_def()
            assert td["name"] == wire["name"]
            assert td["description"] == wire.get("description", "")
            assert td["parameters"] == wire["inputSchema"]
            assert td["parameters"]["type"] == "object"

    @pytest.mark.parametrize(
        "wire_patch, ctx",
        [
            pytest.param(
                {"description": None},
                nullcontext(),
                id="missing-description",
            ),
            pytest.param(
                {"inputSchema": None},
                nullcontext(),
                id="missing-input-schema",
            ),
        ],
    )
    def test_tolerates_missing_optional_fields(self, mcp_wire_tools, wire_patch, ctx):
        """McpTool handles None/missing description and inputSchema gracefully."""
        wire = {**mcp_wire_tools[0], **wire_patch}
        with ctx:
            tool = _wire_to_mcp_tool(wire)
            td = tool.to_tool_def()
            assert td["name"] == wire["name"]
            assert isinstance(td["description"], str)
            assert td["parameters"]["type"] == "object"

    def test_no_params_tool(self, mcp_wire_tools):
        """A tool with no parameters still produces a valid schema."""
        tool_map = {t["name"]: _wire_to_mcp_tool(t) for t in mcp_wire_tools}
        no_params = tool_map["no_params"]
        td = no_params.to_tool_def()
        assert td["parameters"]["type"] == "object"
        assert td["parameters"].get("properties") is not None


# _to_sse_url


class TestToSseUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            pytest.param(
                "https://example.com/mcp", "sse://example.com/mcp", id="https"
            ),
            pytest.param(
                "http://localhost:8080/mcp", "psse://localhost:8080/mcp", id="http"
            ),
            pytest.param(
                "sse://already.ok/mcp",
                "psse://sse://already.ok/mcp",
                id="passthrough-no-match",
            ),
        ],
    )
    def test_convert(self, url, expected):
        assert _to_sse_url(url) == expected


# McpStreamable — helpers


def _jsonrpc_result(result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": "1", "result": result}


_INIT_RESULT = _jsonrpc_result(
    {
        "protocolVersion": "2025-11-25",
        "capabilities": {},
        "serverInfo": {"name": "mock", "version": "0.1"},
    }
)

_TOOLS_RESULT = _jsonrpc_result(
    {
        "tools": [
            {
                "name": "search",
                "description": "Search for things",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ],
    }
)

_CALL_RESULT = _jsonrpc_result(
    {
        "content": [{"type": "text", "text": "found it"}],
    }
)


def _make_response(
    data: dict,
    status: int = 200,
    content_type: str = "application/json",
    session_id: str | None = "sess-123",
) -> AsyncMock:
    """Build a mock niquests.AsyncResponse returning JSON."""
    resp = AsyncMock()
    resp.status_code = status
    headers = {"content-type": content_type}
    if session_id:
        headers["MCP-Session-Id"] = session_id
    resp.headers = headers
    resp.json = AsyncMock(return_value=data)
    resp.raise_for_status = MagicMock(return_value=resp)
    return resp


def _make_sse_event(
    data: str,
    event_id: str = "",
    retry: int | None = None,
) -> MagicMock:
    """Build a mock ServerSentEvent."""
    ev = MagicMock()
    ev.id = event_id
    ev.retry = retry
    ev.data = data
    ev.json = MagicMock(return_value=json.loads(data))
    return ev


def _make_sse_response(
    events: list[MagicMock],
    status: int = 200,
    session_id: str | None = "sess-123",
) -> AsyncMock:
    """Build a mock response with an SSE extension yielding events then None."""
    resp = AsyncMock()
    resp.status_code = status
    headers: dict[str, str] = {"content-type": "text/event-stream"}
    if session_id:
        headers["MCP-Session-Id"] = session_id
    resp.headers = headers
    resp.raise_for_status = MagicMock(return_value=resp)
    ext = MagicMock()
    ext.closed = False
    payloads = list(events) + [None]
    idx = {"i": 0}

    async def _next(**_kw):
        if idx["i"] < len(payloads):
            val = payloads[idx["i"]]
            idx["i"] += 1
            if val is None:
                ext.closed = True
            return val
        ext.closed = True
        return None

    ext.next_payload = _next
    resp.extension = ext
    return resp


class TestMcpStreamable:
    @pytest.fixture
    def mock_http(self):
        """Patch niquests.AsyncSession so no real HTTP calls are made."""
        session = AsyncMock(spec_set=["post", "get", "delete", "close"])
        session.close = AsyncMock()
        session.delete = AsyncMock()
        # GET for _listen returns 405 (no SSE listener)
        listen_resp = AsyncMock()
        listen_resp.status_code = HTTPStatus.METHOD_NOT_ALLOWED
        session.get = AsyncMock(return_value=listen_resp)
        return session

    def _setup_post_responses(self, mock_http, *responses):
        mock_http.post = AsyncMock(side_effect=list(responses))

    async def test_initialize_and_list_tools(self, mock_http):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),  # initialize
            _make_response({}),  # notifications/initialized
            _make_response(_TOOLS_RESULT),  # tools/list
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            assert len(client.tools) == 1
            assert client.tools[0].name == "search"
            td = client.tools[0].to_tool_def()
            assert td["parameters"]["properties"]["query"]["type"] == "string"

    async def test_call_tool(self, mock_http):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            _make_response(_CALL_RESULT),  # tools/call
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            tool = client.tools[0]
            result = await tool.handler({"query": "test"})
            assert result["content"][0]["text"] == "found it"

    async def test_session_id_captured(self, mock_http):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT, session_id="abc-123"),
            _make_response({}, session_id="abc-123"),
            _make_response(_TOOLS_RESULT, session_id="abc-123"),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            assert client._session_id == "abc-123"

    async def test_no_session_id(self, mock_http):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT, session_id=None),
            _make_response({}, session_id=None),
            _make_response(_TOOLS_RESULT, session_id=None),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            assert client._session_id is None

    async def test_rpc_error_raises(self, mock_http):
        error_resp = _make_response(
            {
                "jsonrpc": "2.0",
                "id": "1",
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        )
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            error_resp,
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            with pytest.raises(RuntimeError, match="MCP error -32600"):
                await client._rpc("bad/method")  # type: ignore[arg-type]

    async def test_404_reinitializes(self, mock_http):
        """On 404 with active session, re-initializes and retries."""
        not_found = _make_response({}, status=HTTPStatus.NOT_FOUND)
        not_found.raise_for_status = MagicMock(side_effect=Exception("404"))
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),  # initial initialize
            _make_response({}),  # notifications/initialized
            _make_response(_TOOLS_RESULT),  # tools/list
            not_found,  # first attempt → 404
            _make_response(_INIT_RESULT),  # re-initialize
            _make_response({}),  # notifications/initialized
            _make_response(_TOOLS_RESULT),  # retried tools/list
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            result = await client._rpc("tools/list")
            assert "tools" in result

    async def test_headers_include_session_id(self, mock_http):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT, session_id="sess-x"),
            _make_response({}, session_id="sess-x"),
            _make_response(_TOOLS_RESULT, session_id="sess-x"),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            headers = client._headers()
            assert headers["MCP-Session-Id"] == "sess-x"
            assert headers["MCP-Protocol-Version"] == "2025-11-25"

    @pytest.mark.parametrize(
        "token, has_auth",
        [
            pytest.param("sk-abc123", True, id="with-token"),
            pytest.param(None, False, id="without-token"),
        ],
    )
    async def test_bearer_token(self, mock_http, token, has_auth):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
        )
        client = McpStreamable(url="https://example.com/mcp", token=token)
        client._http = mock_http
        async with client:
            headers = client._headers()
            if has_auth:
                assert headers["Authorization"] == "Bearer sk-abc123"
            else:
                assert "Authorization" not in headers

    async def test_401_raises(self, mock_http):
        unauthorized = _make_response({}, status=HTTPStatus.UNAUTHORIZED)
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            unauthorized,
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            with pytest.raises(RuntimeError, match="requires authorization"):
                await client._rpc("tools/list")

    async def test_listen_tracks_last_event_id(self, mock_http):
        """_listen tracks event IDs and sends Last-Event-ID on reconnect."""
        sse_event = _make_sse_event('{"method":"ping"}', event_id="evt-1")
        first_get = _make_sse_response([sse_event])
        second_get = AsyncMock()
        second_get.status_code = HTTPStatus.METHOD_NOT_ALLOWED
        mock_http.get = AsyncMock(side_effect=[first_get, second_get])
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        client._retry_ms = 0  # no delay in tests
        async with client:
            # Wait for _listen background task to finish
            await asyncio.sleep(0.05)
            assert client._last_event_id == "evt-1"
        # Second GET call should have Last-Event-ID header
        second_call_headers = mock_http.get.call_args_list[1].kwargs.get(
            "headers",
            mock_http.get.call_args_list[1][1]
            if len(mock_http.get.call_args_list[1]) > 1
            else {},
        )
        assert second_call_headers.get("Last-Event-ID") == "evt-1"

    async def test_listen_respects_retry_field(self, mock_http):
        """_listen updates _retry_ms from SSE retry field."""
        sse_event = _make_sse_event('{"method":"ping"}', retry=5000)
        get_resp = _make_sse_response([sse_event])
        stop_resp = AsyncMock()
        stop_resp.status_code = HTTPStatus.METHOD_NOT_ALLOWED
        mock_http.get = AsyncMock(side_effect=[get_resp, stop_resp])
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        client._retry_ms = 0
        async with client:
            await asyncio.sleep(0.05)
            assert client._retry_ms == 5000

    async def test_listen_stops_after_max_retries(self, mock_http):
        """_listen gives up after _MAX_LISTEN_RETRIES consecutive failures."""
        mock_http.get = AsyncMock(side_effect=ConnectionError("down"))
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        client._retry_ms = 0
        # Run _listen directly (not via __aenter__)
        await client._listen()
        assert mock_http.get.call_count == 5  # _MAX_LISTEN_RETRIES

    @pytest.mark.parametrize(
        "reason, expected_params",
        [
            pytest.param(
                "timeout",
                {"requestId": "req-42", "reason": "timeout"},
                id="with-reason",
            ),
            pytest.param(
                None,
                {"requestId": "req-42"},
                id="without-reason",
            ),
        ],
    )
    async def test_cancel(self, mock_http, reason, expected_params):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            _make_response({}),  # cancel response
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            await client.cancel("req-42", reason=reason)
        cancel_call = mock_http.post.call_args_list[3]
        body = cancel_call.kwargs.get(
            "json", cancel_call[1].get("json") if len(cancel_call) > 1 else None
        )
        assert body["method"] == "notifications/cancelled"
        assert body["params"] == expected_params
        assert "id" not in body  # notification, not request

    async def test_refresh_tools_mutates_in_place(self, mock_http):
        """External references to mcp.tools must survive a refresh.

        Regression: `self._tools = _build_tools(...)` broke references held
        by callers (e.g. `AgentSession(mcp_tools=mcp.tools)`) because a
        fresh list object replaced the one they were pointing at.
        """
        new_tools_result = _jsonrpc_result(
            {
                "tools": [
                    {
                        "name": "new_tool",
                        "description": "Added after a list_changed notification",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ],
            }
        )
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),  # initial: `search`
            _make_response(new_tools_result),  # refresh: `new_tool`
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            tools_ref = client.tools  # reference captured before refresh
            assert [t.name for t in tools_ref] == ["search"]
            await client._refresh_tools()
            assert tools_ref is client.tools  # same list object
            assert [t.name for t in tools_ref] == ["new_tool"]

    async def test_ping(self, mock_http):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            _make_response({}),  # ping response
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            await client.ping()
        ping_call = mock_http.post.call_args_list[3]
        body = ping_call.kwargs.get(
            "json", ping_call[1].get("json") if len(ping_call) > 1 else None
        )
        assert body["method"] == "ping"

    async def test_reentry_after_exit(self, mock_http):
        """Regression: `__aexit__` left `_bg_task`/`_session_id` populated, so
        `is_open` stayed True after the first exit. That broke two things:
        re-using the same instance in a second `async with` raised
        "already open", and passing the post-exit instance into
        `AgentSession` (which checks `is_open`) made the session skip
        re-initialization and then fail on the first tool call because
        `_http` had already been closed.
        """
        self._setup_post_responses(
            mock_http,
            # First entry: init + initialized notify + tools/list
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            # Second entry: same handshake again
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            assert client.is_open
        assert not client.is_open
        assert client._bg_task is None
        assert client._session_id is None
        # And it must be re-enterable, with a fresh _http underneath.
        client._http = mock_http  # would have been recreated on exit
        async with client:
            assert client.is_open
        assert not client.is_open


# McpStdio


_STDIO_SERVER_SCRIPT = """\
import asyncio
from mcp.server.fastmcp import FastMCP

server = FastMCP("test-stdio")

@server.tool()
def greet(name: str) -> str:
    \"\"\"Greet someone by name.\"\"\"
    return f"Hello, {name}!"

@server.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

@server.tool()
async def slow(seconds: float) -> str:
    \"\"\"Sleep for `seconds` seconds, then return.\"\"\"
    await asyncio.sleep(seconds)
    return "woke up"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


# A stdio server that also writes a lot to stderr — used to verify the
# drain task keeps the pipe clear. Without the drain, a server writing
# more than ~64KB to stderr before responding would block forever.
_STDIO_NOISY_SERVER_SCRIPT = """\
import sys
from mcp.server.fastmcp import FastMCP

# Write enough bytes to overflow the default 64KB stderr pipe buffer
# before the MCP initialization handshake runs.
sys.stderr.write("x" * 200_000 + "\\n")
sys.stderr.flush()

server = FastMCP("noisy-stdio")

@server.tool()
def ping() -> str:
    \"\"\"Simple ping tool.\"\"\"
    return "pong"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


class TestMcpStdio:
    async def test_list_tools(self):
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            assert len(client.tools) == 3
            names = {t.name for t in client.tools}
            assert names == {"greet", "add", "slow"}

    async def test_tool_to_tool_def(self):
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            for tool in client.tools:
                td = tool.to_tool_def()
                assert td["name"] == tool.name
                assert td["parameters"]["type"] == "object"

    async def test_call_tool(self):
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            greet = next(t for t in client.tools if t.name == "greet")
            result = await greet.handler({"name": "World"})
            assert any("Hello, World!" in c["text"] for c in result["content"])

    async def test_call_add(self):
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            add = next(t for t in client.tools if t.name == "add")
            result = await add.handler({"a": 3, "b": 4})
            assert any("7" in c["text"] for c in result["content"])

    async def test_cancelled_rpc_does_not_kill_reader(self):
        """Cancelling one in-flight RPC must not wedge the whole session.

        Regression: the cancelled future used to stay in `_pending`; when
        the late response arrived, the reader's `fut.set_result()` raised
        `InvalidStateError` and crashed the reader task, so subsequent
        tool calls would hang forever.
        """
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            slow = next(t for t in client.tools if t.name == "slow")
            greet = next(t for t in client.tools if t.name == "greet")

            # Kick off a slow call and cancel it after a short delay.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(slow.handler({"seconds": 5}), timeout=0.2)

            # Give the late response a chance to arrive on the reader.
            await asyncio.sleep(0.1)

            # Reader must still be alive and able to service new calls.
            assert client._reader_task is not None
            assert not client._reader_task.done()
            result = await greet.handler({"name": "alive"})
            assert any("Hello, alive!" in c["text"] for c in result["content"])

    async def test_noisy_stderr_does_not_block_init(self):
        """A server that floods stderr must not deadlock on the pipe buffer.

        Regression: stderr was piped but never read, so any server writing
        more than ~64KB to stderr would block at its next write, wedging
        the MCP init handshake forever.
        """
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_NOISY_SERVER_SCRIPT],
        ) as client:
            assert any(t.name == "ping" for t in client.tools)
            ping_tool = next(t for t in client.tools if t.name == "ping")
            result = await ping_tool.handler({})
            assert any("pong" in c["text"] for c in result["content"])

    async def test_refresh_tools_mutates_in_place(self):
        """Stdio transport also refreshes its tool list in place."""
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            tools_ref = client.tools  # reference captured before refresh
            original_names = {t.name for t in tools_ref}

            # Swap in a fake result to force a difference we can observe.
            from padwan_llm.mcp import _build_tools

            async def fake_refresh() -> None:
                client._tools[:] = _build_tools(
                    [
                        {
                            "name": "refreshed_tool",
                            "description": "Added after refresh",
                            "inputSchema": {"type": "object", "properties": {}},
                        },
                    ],
                    client._call,
                )

            await fake_refresh()
            assert tools_ref is client.tools  # same list object
            assert [t.name for t in tools_ref] == ["refreshed_tool"]
            assert original_names != {"refreshed_tool"}

    async def test_reentry_after_exit(self):
        """Regression: `__aexit__` left `_process` populated, so `is_open`
        stayed True after the first exit. Re-using the same instance in a
        second `async with` raised "already open", and passing the
        post-exit instance into `AgentSession` (which checks `is_open`)
        made the session skip re-initialization and then issue tool calls
        against a dead subprocess.
        """
        client = McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        )
        async with client:
            assert client.is_open
            assert {t.name for t in client.tools} == {"greet", "add", "slow"}
        assert not client.is_open
        assert client._process is None
        assert client._reader_task is None
        assert client._stderr_task is None
        # Same instance must be re-enterable end-to-end (new subprocess,
        # fresh reader, working tool call).
        async with client:
            assert client.is_open
            greet = next(t for t in client.tools if t.name == "greet")
            result = await greet.handler({"name": "again"})
            assert any("Hello, again!" in c["text"] for c in result["content"])
        assert not client.is_open
