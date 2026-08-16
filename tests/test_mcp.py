import asyncio
import json
import sys
from contextlib import nullcontext
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.mcpserver import MCPServer

from padwan_llm.mcp import (
    McpStdio,
    McpStreamable,
    McpTool,
    _build_tools,
    _normalize_call_result,
    _sanitize_prefix,
    _to_sse_url,
)


def _make_server() -> MCPServer:
    server = MCPServer("test")

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
    """Wire-format tool dicts as returned by a real mcp MCPServer server."""
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
    def test_round_trip_fields_and_tool_defs(self, mcp_wire_tools):
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

        for wire in mcp_wire_tools:
            tool = _wire_to_mcp_tool(wire)
            td = tool.to_tool_def()
            assert td["name"] == wire["name"]
            assert td["description"] == wire.get("description", "")
            assert td["parameters"] == wire["inputSchema"]
            assert td["parameters"]["type"] == "object"
        no_params = tool_map["no_params"]
        td = no_params.to_tool_def()
        assert td["parameters"]["type"] == "object"
        assert td["parameters"].get("properties") is not None

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


@pytest.mark.parametrize(
    "raw_in, expected",
    [
        pytest.param("simple", "simple", id="passthrough"),
        pytest.param("mcp.data.gouv.fr", "mcp_data_gouv_fr", id="dots_to_underscore"),
        pytest.param(
            "server-filesystem", "server_filesystem", id="dashes_to_underscore"
        ),
        pytest.param("__weird__", "weird", id="strip_leading_trailing_underscores"),
        pytest.param("", "mcp", id="empty_falls_back_to_mcp"),
        pytest.param("///", "mcp", id="all_invalid_falls_back"),
        pytest.param("a.b/c-d:e", "a_b_c_d_e", id="mixed_separators"),
    ],
)
def test_sanitize_prefix(raw_in: str, expected: str):
    """`_sanitize_prefix` reduces arbitrary host/command strings to a
    function-name-safe identifier (matches the OpenAI/Gemini/Anthropic
    `[a-zA-Z0-9_-]` regex). Empty input falls back to ``"mcp"`` so the
    auto-prefix path always produces something usable."""
    assert _sanitize_prefix(raw_in) == expected


@pytest.mark.parametrize(
    "url, expected",
    [
        pytest.param(
            "https://mcp.data.gouv.fr/mcp", "mcp_data_gouv_fr", id="dotted_host"
        ),
        pytest.param("http://localhost:8080/mcp", "localhost", id="localhost"),
        pytest.param(
            "https://api.weather.example.com/v1/mcp",
            "api_weather_example_com",
            id="subdomain",
        ),
    ],
)
def test_streamable_auto_prefix(url, expected):
    """`McpStreamable.auto_prefix` derives a stable, readable namespace
    from the URL host so collision-driven auto-prefixing produces
    something the model can actually use in prompts."""
    assert McpStreamable(url=url).auto_prefix == expected


@pytest.mark.parametrize(
    "command, args, expected",
    [
        pytest.param(
            "npx",
            ["@modelcontextprotocol/server-filesystem", "/path"],
            "server_filesystem",
            id="strips_npm_scope",
        ),
        pytest.param("python", ["main.py"], "main", id="strips_py_extension"),
        pytest.param("node", ["index.js"], "index", id="strips_js_extension"),
        pytest.param("npx", [], "npx", id="empty_args_falls_back_to_command"),
        pytest.param(
            "/usr/bin/python", ["server.py"], "server", id="basename_strips_py"
        ),
        pytest.param(
            "node",
            ["/abs/path/to/server-foo.mjs"],
            "server_foo",
            id="basename_strips_mjs_and_dashes",
        ),
    ],
)
def test_stdio_auto_prefix(command, args, expected):
    """`McpStdio.auto_prefix` prefers `args[0]` (the script/package
    being launched) over the launcher binary, since `npx`/`python` are
    rarely meaningful identifiers on their own."""
    assert McpStdio(command=command, args=args).auto_prefix == expected


def test_streamable_label_is_url():
    """`McpStreamable.label` exposes the configured URL verbatim so
    callers can show the transport in notifications/logs without
    reaching into URL parsing or the sanitized `auto_prefix`."""
    url = "https://mcp.data.gouv.fr/mcp"
    assert McpStreamable(url=url).label == url


@pytest.mark.parametrize(
    "command, args, expected",
    [
        pytest.param(
            "npx",
            ["@modelcontextprotocol/server-filesystem", "/path"],
            "npx @modelcontextprotocol/server-filesystem /path",
            id="full_command_line",
        ),
        pytest.param("python", ["main.py"], "python main.py", id="single_arg"),
        pytest.param("npx", [], "npx", id="empty_args_is_command_only"),
    ],
)
def test_stdio_label(command, args, expected):
    """`McpStdio.label` returns the full command line as the caller
    configured it — unlike `auto_prefix`, nothing is stripped or
    sanitized, so the label is suitable for display."""
    assert McpStdio(command=command, args=args).label == expected


@pytest.mark.parametrize(
    "name_prefix, expected_names",
    [
        pytest.param(None, ["forecast", "alerts"], id="no_prefix_uses_wire_names"),
        pytest.param(
            "weather",
            ["weather__forecast", "weather__alerts"],
            id="explicit_prefix_applied",
        ),
    ],
)
def test_build_tools_applies_name_prefix(
    name_prefix: str | None, expected_names: list[str]
):
    """With `name_prefix=None`, tools keep their wire names. With an
    explicit prefix, every tool is renamed to `<prefix>__<wire>` while
    the handler stays bound to the underlying wire name so the
    tools/call dispatch still hits the right backend method.
    """

    async def fake_call(name, args):
        return {"called": name, "args": args}

    raw = [
        {"name": "forecast", "description": "Get forecast", "inputSchema": {}},
        {"name": "alerts", "description": "Get alerts", "inputSchema": {}},
    ]
    built = _build_tools(raw, fake_call, name_prefix=name_prefix)
    assert [t.name for t in built] == expected_names
    # Handler must always call the underlying wire name, not the prefixed one.
    result = asyncio.run(built[0].handler({"city": "Paris"}))
    assert result == {"called": "forecast", "args": {"city": "Paris"}}


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param(
            {"content": [{"type": "text", "text": "hello"}]},
            {"content": [{"type": "text", "text": "hello"}]},
            id="text_only_passthrough",
        ),
        pytest.param(
            {
                "content": [
                    {"type": "text", "text": "summary"},
                    {"type": "image", "data": "base64...", "mimeType": "image/png"},
                    {"type": "resource", "resource": {"uri": "file://x"}},
                ]
            },
            {
                "content": [
                    {"type": "text", "text": "summary"},
                    {"type": "image", "data": "base64...", "mimeType": "image/png"},
                    {"type": "resource", "resource": {"uri": "file://x"}},
                ]
            },
            id="non_text_blocks_preserved",
        ),
        pytest.param(
            {"structuredContent": {"answer": 42}, "content": []},
            {"structuredContent": {"answer": 42}, "content": []},
            id="structured_content_preserved",
        ),
        pytest.param(
            {"content": [{"type": "text", "text": "boom"}], "isError": True},
            {"content": [{"type": "text", "text": "boom"}], "isError": True},
            id="is_error_flag_preserved",
        ),
        pytest.param({"content": []}, {"content": []}, id="empty_content"),
        pytest.param(None, {"content": []}, id="none_normalised"),
        pytest.param("not a dict", {"content": []}, id="non_dict_normalised"),
    ],
)
def test_normalize_call_result(result, expected):
    """Regression: `_extract_text_content` used to rebuild the result from
    text blocks only, silently dropping `structuredContent`, `isError`,
    and any image/resource blocks. JSON-only MCP servers were rendered
    unusable because the model never saw the actual tool output. Now
    we pass the dict through unchanged.
    """
    assert _normalize_call_result(result) == expected


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
    """Build a mock ServerSentEvent.

    Empty `data` represents an SSE keep-alive frame; `event.json()` is
    wired to raise `ValueError` so the test fails loudly if production
    code accidentally tries to parse it instead of skipping it.
    """
    ev = MagicMock()
    ev.id = event_id
    ev.retry = retry
    ev.data = data
    if data:
        ev.json = MagicMock(return_value=json.loads(data))
    else:
        ev.json = MagicMock(side_effect=ValueError("empty SSE data"))
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
            assert len(client.tools) == 1
            tool = client.tools[0]
            assert tool.name == "search"
            td = tool.to_tool_def()
            assert td["parameters"]["properties"]["query"]["type"] == "string"
            result = await tool.handler({"query": "test"})
            assert result["content"][0]["text"] == "found it"

    @pytest.mark.parametrize(
        "session_id, expected_header_value",
        [
            pytest.param("abc-123", "abc-123", id="session_id_captured"),
            pytest.param(None, None, id="no_session_id"),
        ],
    )
    async def test_session_id_state(
        self,
        mock_http,
        session_id,
        expected_header_value,
    ):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT, session_id=session_id),
            _make_response({}, session_id=session_id),
            _make_response(_TOOLS_RESULT, session_id=session_id),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            assert client._session_id == session_id
            headers = client._headers()
            assert headers["MCP-Protocol-Version"] == "2025-11-25"
            if expected_header_value is None:
                assert "MCP-Session-Id" not in headers
            else:
                assert headers["MCP-Session-Id"] == expected_header_value

    @pytest.mark.parametrize(
        "error_resp, expected_match",
        [
            pytest.param(
                _make_response(
                    {
                        "jsonrpc": "2.0",
                        "id": "1",
                        "error": {"code": -32600, "message": "Invalid Request"},
                    }
                ),
                "MCP error -32600",
                id="jsonrpc_error",
            ),
            pytest.param(
                _make_response({}, status=HTTPStatus.UNAUTHORIZED),
                "requires authorization",
                id="unauthorized",
            ),
        ],
    )
    async def test_rpc_failures_raise(self, mock_http, error_resp, expected_match):
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
            with pytest.raises(RuntimeError, match=expected_match):
                await client._rpc("bad/method")  # type: ignore[arg-type]

    async def test_404_reinitializes(self, mock_http):
        """On 404 with active session, re-initializes and retries."""
        not_found = _make_response({}, status=HTTPStatus.NOT_FOUND)
        not_found.raise_for_status = MagicMock(side_effect=Exception("404"))
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            not_found,  # first attempt → 404
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            result = await client._rpc("tools/list")
            assert "tools" in result

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

    async def test_listen_skips_sse_keepalives(self, mock_http, caplog):
        """Empty `data:` SSE frames are keep-alives, not malformed JSON.

        Regression: `_listen()` used to call `event.json()` unconditionally
        and only catch `ValueError` afterwards, logging every keep-alive
        as `MCP: ignoring malformed SSE event:` — flooding the chat UI
        on servers like https://mcp.data.gouv.fr/mcp that emit them
        every few seconds.
        """
        keepalive = _make_sse_event("")
        real = _make_sse_event('{"method":"ping"}')
        get_resp = _make_sse_response([keepalive, real])
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
        with caplog.at_level("WARNING", logger="padwan_llm"):
            async with client:
                await asyncio.sleep(0.05)
        # The keep-alive must NOT have triggered event.json() at all,
        # and must NOT have produced a "malformed SSE event" warning.
        keepalive.json.assert_not_called()
        assert not any(
            "malformed SSE event" in rec.message for rec in caplog.records
        ), [rec.message for rec in caplog.records]

    async def test_read_sse_response_skips_keepalives(self, mock_http):
        """Same fix on the foreground RPC channel.

        Regression: `_read_sse_response` called `event.json()` with no
        error handling at all, so a stray keep-alive arriving between
        notifications and the actual JSON-RPC reply would crash the RPC.
        """
        keepalive = _make_sse_event("")
        rpc_reply = _make_sse_event(
            json.dumps({"jsonrpc": "2.0", "id": "1", "result": {"tools": []}})
        )
        sse_resp = AsyncMock()
        sse_resp.status_code = 200
        sse_resp.headers = {
            "content-type": "text/event-stream",
            "MCP-Session-Id": "sess-1",
        }
        sse_resp.raise_for_status = MagicMock(return_value=sse_resp)
        ext = MagicMock()
        ext.closed = False
        payloads = [keepalive, rpc_reply, None]
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
        sse_resp.extension = ext

        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            sse_resp,  # the tools/list call returns SSE with a keep-alive
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        async with client:
            pass  # _refresh_tools (called from __aenter__) drives _read_sse_response
        # No exception means the keep-alive was skipped before json().
        keepalive.json.assert_not_called()

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

    async def test_aenter_cleans_up_on_failed_refresh_tools(self, mock_http):
        """Regression: if `_initialize` succeeds but `_refresh_tools` raises
        (e.g. server returns a JSON-RPC error on `tools/list`), the
        transport used to leak: `_session_id` stayed populated, the HTTP
        session stayed open, and `is_open` stayed False but the next
        retry would carry a stale session id into a new `_initialize`.
        Now `__aenter__` runs the cleanup path before re-raising.
        """
        # init succeeds, initialized notification succeeds, tools/list fails
        broken_tools_resp = _make_response(
            {
                "jsonrpc": "2.0",
                "id": "x",
                "error": {"code": -32000, "message": "tools listing broken"},
            }
        )
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            broken_tools_resp,
        )
        client = McpStreamable(url="https://example.com/mcp")
        client._http = mock_http
        original_http = client._http

        with pytest.raises(RuntimeError, match="tools listing broken"):
            await client.__aenter__()

        # State must be fully reset — same shape `__aexit__` leaves behind.
        assert client._bg_task is None
        assert client._session_id is None
        assert client._last_event_id is None
        assert not client.is_open
        # The HTTP session that we partially used must have been replaced
        # with a fresh one (so a retry doesn't reuse a half-closed one).
        assert client._http is not original_http

    @pytest.mark.parametrize(
        "on_auth, extra_responses, expected, check_token",
        [
            pytest.param(
                lambda _t: "new-token",
                [_make_response(_jsonrpc_result({"tools": []}))],
                nullcontext(),
                "new-token",
                id="sync-callback-retries",
            ),
            pytest.param(
                None,
                [],
                pytest.raises(RuntimeError, match="requires authorization"),
                None,
                id="no-callback-raises",
            ),
            pytest.param(
                lambda _t: (_ for _ in ()).throw(ValueError("cancelled")),
                [],
                pytest.raises(ValueError, match="cancelled"),
                None,
                id="callback-raises-propagates",
            ),
            pytest.param(
                lambda _t: "token",
                [_make_response({}, status=HTTPStatus.UNAUTHORIZED)],
                pytest.raises(RuntimeError, match="requires authorization"),
                None,
                id="double-401-retries-only-once",
            ),
        ],
    )
    async def test_401_recovery(
        self, mock_http, on_auth, extra_responses, expected, check_token
    ):
        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            _make_response({}, status=HTTPStatus.UNAUTHORIZED),
            *extra_responses,
        )
        client = McpStreamable(url="https://example.com/mcp", on_auth=on_auth)
        client._http = mock_http
        async with client:
            with expected:
                await client._rpc("tools/list")
            if check_token is not None:
                assert client.token == check_token

    async def test_401_async_on_auth(self, mock_http):
        async def _async_auth(_transport):
            return "async-token"

        self._setup_post_responses(
            mock_http,
            _make_response(_INIT_RESULT),
            _make_response({}),
            _make_response(_TOOLS_RESULT),
            _make_response({}, status=HTTPStatus.UNAUTHORIZED),
            _make_response(_jsonrpc_result({"tools": []})),
        )
        client = McpStreamable(url="https://example.com/mcp", on_auth=_async_auth)
        client._http = mock_http
        async with client:
            await client._rpc("tools/list")
            assert client.token == "async-token"


# McpStdio


_STDIO_SERVER_SCRIPT = """\
import asyncio
from mcp.server.mcpserver import MCPServer

server = MCPServer("test-stdio")

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
from mcp.server.mcpserver import MCPServer

# Write enough bytes to overflow the default 64KB stderr pipe buffer
# before the MCP initialization handshake runs.
sys.stderr.write("x" * 200_000 + "\\n")
sys.stderr.flush()

server = MCPServer("noisy-stdio")

@server.tool()
def ping() -> str:
    \"\"\"Simple ping tool.\"\"\"
    return "pong"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


class TestMcpStdio:
    async def test_list_tools_and_tool_defs(self):
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            assert len(client.tools) == 3
            names = {t.name for t in client.tools}
            assert names == {"greet", "add", "slow"}
            for tool in client.tools:
                td = tool.to_tool_def()
                assert td["name"] == tool.name
                assert td["parameters"]["type"] == "object"

    @pytest.mark.parametrize(
        "tool_name, args, expected_text",
        [
            pytest.param(
                "greet",
                {"name": "World"},
                "Hello, World!",
                id="greet_tool",
            ),
            pytest.param("add", {"a": 3, "b": 4}, "7", id="add_tool"),
        ],
    )
    async def test_call_tool_returns_content(
        self,
        tool_name: str,
        args: dict[str, int | str],
        expected_text: str,
    ) -> None:
        async with McpStdio(
            command=sys.executable,
            args=["-c", _STDIO_SERVER_SCRIPT],
        ) as client:
            tool = next(t for t in client.tools if t.name == tool_name)
            result = await tool.handler(args)
            assert any(expected_text in c["text"] for c in result["content"])

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

    async def test_aexit_reaps_process_after_sigkill(self):
        """Regression: after `kill()`, `__aexit__` used to clear `_process`
        without awaiting `wait()`. POSIX requires the parent to call
        `wait()` to reap the zombie — without it, every wedged child
        becomes a defunct process the parent never collects, and
        recycling stdio transports leaks them. We mock the process so
        the SIGKILL path runs deterministically: `wait()` raises
        `TimeoutError` directly (so `asyncio.wait_for` re-raises it
        immediately, no real 2 s sleeps) until `kill()` is called.
        """
        wait_calls = {"count": 0}
        kill_called = {"value": False}

        async def fake_wait():
            wait_calls["count"] += 1
            if not kill_called["value"]:
                # `wait_for` propagates exceptions from the inner coro,
                # so raising TimeoutError here looks identical to the
                # real timeout firing — but instantly.
                raise asyncio.TimeoutError()
            return 0

        proc = MagicMock()
        proc.wait = fake_wait
        proc.terminate = MagicMock()
        proc.kill = MagicMock(side_effect=lambda: kill_called.update(value=True))

        client = McpStdio(command="anything", args=[])
        client._process = proc
        # Pretend the reader/stderr tasks were never created so
        # __aexit__'s task-cleanup branches no-op.
        client._reader_task = None
        client._stderr_task = None

        await client.__aexit__()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        # Three wait_for(wait()) calls: clean exit, post-terminate,
        # post-kill. The third one MUST happen — it's the regression.
        assert wait_calls["count"] == 3
        # And state must be reset just like a normal exit.
        assert client._process is None

    async def test_aenter_cleans_up_on_failed_initialize(self):
        """Regression: a server that crashes during the MCP handshake used
        to leave the child process and the reader/stderr tasks running.
        `__aexit__` would never run because `__aenter__` was raising,
        so `_process`/`_reader_task`/`_stderr_task` stayed populated and
        `is_open` stayed True on a transport that never actually
        initialized — leaking subprocesses across retries.
        """
        # Server that exits immediately so `_initialize` cannot complete.
        bad_script = "import sys\nsys.stderr.write('boom\\n')\nsys.exit(1)\n"
        client = McpStdio(command=sys.executable, args=["-c", bad_script])

        with pytest.raises(Exception):
            await client.__aenter__()

        # State must be fully reset just like a clean `__aexit__` leaves it.
        assert not client.is_open
        assert client._process is None
        assert client._reader_task is None
        assert client._stderr_task is None
