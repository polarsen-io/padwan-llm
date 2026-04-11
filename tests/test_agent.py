import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from padwan_llm import (
    AgentSession,
    ChatMessage,
    ChatStream,
    ConversationSnapshot,
    ConversationState,
    LLMClientBase,
    McpTool,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    UsageToken,
)
from padwan_llm.agent import ToolErrorHandler, _extract_text


# Fakes


@dataclass
class FakeChatStream(ChatStream):
    """Scripted ChatStream — yields preset chunks, exposes preset usage and tool_calls."""

    chunks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] | None = None
    usage: UsageToken | None = None

    async def __aiter__(self) -> AsyncIterator[str]:
        for c in self.chunks:
            yield c


@dataclass
class FakeClient:
    """Returns scripted FakeChatStream responses in order, recording each call."""

    responses: list[FakeChatStream]
    calls: list[tuple[list[ChatMessage], list[ToolDefinition]]] = field(
        default_factory=list
    )
    aenter_count: int = 0
    aexit_count: int = 0
    # Track state so AgentSession can detect whether the fake is "open"
    _is_open: bool = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> ChatStream:
        self.calls.append((list(messages), list(tools or [])))
        return self.responses.pop(0)

    async def __aenter__(self) -> "FakeClient":
        self.aenter_count += 1
        self._is_open = True
        return self

    async def __aexit__(self, *_: object) -> None:
        self.aexit_count += 1
        self._is_open = False


def make_session(
    responses: list[FakeChatStream],
    *,
    mcp_tools: Sequence[McpTool] = (),
    **kwargs: Any,
) -> tuple[AgentSession, FakeClient]:
    client = FakeClient(responses=list(responses))
    # Pass mcp_tools through by reference so tests can mutate the same list
    # the session reads each round.
    session = AgentSession(
        client=cast(LLMClientBase, client),
        mcp_tools=mcp_tools,
        **kwargs,
    )
    return session, client


def make_tool_call(
    name: str, args: dict[str, Any], call_id: str = "call_1"
) -> ToolCall:
    return ToolCall(
        id=call_id,
        type="function",
        function=ToolCallFunction(name=name, arguments=json.dumps(args)),
    )


class FakeStore:
    """In-memory ConversationStore used by every persistence test."""

    def __init__(self) -> None:
        self.storage: dict[str, ConversationSnapshot] = {}

    def save(self, session_id: str, snapshot: ConversationSnapshot) -> None:
        self.storage[session_id] = snapshot

    def load(self, session_id: str) -> ConversationSnapshot:
        return self.storage[session_id]


# Basic loop behaviour


@pytest.mark.parametrize(
    "chunks, expected",
    [
        pytest.param(["Hello", " world"], "Hello world", id="text_response"),
        pytest.param([], "(no response)", id="empty_response"),
    ],
)
async def test_single_round_response(chunks: list[str], expected: str) -> None:
    session, client = make_session([FakeChatStream(chunks=chunks)])
    out = await session.send("hi")
    assert out == expected
    assert len(client.calls) == 1
    assert session.messages[-1] == {"role": "assistant", "content": expected}


async def test_tool_call_round_trip() -> None:
    async def get_weather(args: dict[str, Any]) -> str:
        return f"Sunny in {args['city']}"

    weather = McpTool(
        name="get_weather",
        description="weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        handler=get_weather,
    )
    session, client = make_session(
        [
            FakeChatStream(
                chunks=[],
                tool_calls=[make_tool_call("get_weather", {"city": "Paris"})],
            ),
            FakeChatStream(chunks=["The weather in Paris is sunny."]),
        ],
        mcp_tools=[weather],
    )
    out = await session.send("weather in Paris?")
    assert out == "The weather in Paris is sunny."
    assert len(client.calls) == 2
    # Second call to the LLM should include the tool result
    second_messages = client.calls[1][0]
    tool_msgs = [m for m in second_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "Sunny in Paris"  # type: ignore[typeddict-item]


# Error / unknown tool handling


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param("unknown", id="unknown_tool"),
        pytest.param("raise_default", id="handler_raises_default_error"),
        pytest.param("raise_override", id="handler_raises_with_on_tool_error"),
    ],
)
async def test_tool_failure_modes(scenario: str) -> None:
    seen_errors: list[tuple[str, Exception]] = []

    async def boom(_args: dict[str, Any]) -> str:
        raise RuntimeError("kaboom")

    tools: list[McpTool] = []
    if scenario != "unknown":
        tools.append(
            McpTool(
                name="boom",
                description="",
                input_schema={"type": "object", "properties": {}},
                handler=boom,
            )
        )

    def custom_error(tool: McpTool, args: dict[str, Any], exc: Exception) -> str:
        seen_errors.append((tool.name, exc))
        return "custom error message"

    on_tool_error: ToolErrorHandler | None = (
        custom_error if scenario == "raise_override" else None
    )

    name = "missing" if scenario == "unknown" else "boom"
    session, client = make_session(
        [
            FakeChatStream(chunks=[], tool_calls=[make_tool_call(name, {})]),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=tools,
        on_tool_error=on_tool_error,
    )
    out = await session.send("go")
    assert out == "done"

    second_messages = client.calls[1][0]
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    content = cast(str, tool_msg["content"])  # type: ignore[typeddict-item]

    if scenario == "unknown":
        assert json.loads(content) == {"error": "Unknown tool: missing"}
    elif scenario == "raise_default":
        assert json.loads(content) == {"error": "kaboom"}
    else:
        assert content == "custom error message"
        assert seen_errors == [("boom", seen_errors[0][1])]
        assert isinstance(seen_errors[0][1], RuntimeError)


# Round limit


async def test_max_tool_rounds_limits_llm_calls() -> None:
    async def looper(_args: dict[str, Any]) -> str:
        return "still working"

    tool = McpTool(
        name="loop",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=looper,
    )
    # The LLM keeps calling the tool — we cap it at 1 LLM call
    responses = [
        FakeChatStream(
            chunks=[],
            tool_calls=[make_tool_call("loop", {}, call_id=f"c{i}")],
        )
        for i in range(5)
    ]
    session, client = make_session(responses, mcp_tools=[tool], max_tool_rounds=1)
    out = await session.send("go")
    assert "reached tool call limit of 1 rounds" in out
    assert len(client.calls) == 1


def test_invalid_max_tool_rounds_rejected() -> None:
    with pytest.raises(ValueError, match="max_tool_rounds"):
        AgentSession(client=cast(LLMClientBase, FakeClient([])), max_tool_rounds=0)


# Context truncation


@pytest.mark.parametrize(
    "max_tool_result_chars, expected_truncated",
    [
        pytest.param(100, True, id="limit_truncates"),
        pytest.param(None, False, id="none_disables"),
    ],
)
async def test_tool_result_context_truncation(
    max_tool_result_chars: int | None, expected_truncated: bool
) -> None:
    huge = "x" * 50_000

    async def big(_args: dict[str, Any]) -> str:
        return huge

    tool = McpTool(
        name="big",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=big,
    )
    session, client = make_session(
        [
            FakeChatStream(chunks=[], tool_calls=[make_tool_call("big", {})]),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=[tool],
        max_tool_result_chars=max_tool_result_chars,
    )
    await session.send("go")

    # Stored history always keeps the full result
    stored = next(m for m in session.messages if m.get("role") == "tool")
    assert len(stored["content"]) == 50_000  # type: ignore[typeddict-item]

    second_messages = client.calls[1][0]
    sent = next(m for m in second_messages if m.get("role") == "tool")
    if expected_truncated:
        assert len(sent["content"]) <= 200  # type: ignore[typeddict-item]
        assert sent["content"].endswith("[truncated]")  # type: ignore[typeddict-item]
    else:
        assert len(sent["content"]) == 50_000  # type: ignore[typeddict-item]


# Hooks


async def test_on_tool_callback_receives_name_and_args() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def handler(args: dict[str, Any]) -> str:
        return "ok"

    tool = McpTool(
        name="t",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )
    session, _ = make_session(
        [
            FakeChatStream(chunks=[], tool_calls=[make_tool_call("t", {"k": 1})]),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=[tool],
        on_tool=lambda name, args: calls.append((name, args)),
    )
    await session.send("go")
    assert calls == [("t", {"k": 1})]


def _sync_deny(_t: McpTool, _a: dict[str, Any]) -> bool:
    return False


async def _async_deny(_t: McpTool, _a: dict[str, Any]) -> bool:
    return False


@pytest.mark.parametrize(
    "hook",
    [
        pytest.param(_sync_deny, id="sync_hook"),
        pytest.param(_async_deny, id="async_hook"),
    ],
)
async def test_approve_tool_denied_skips_handler(hook: Any) -> None:
    invoked = False

    async def handler(_args: dict[str, Any]) -> str:
        nonlocal invoked
        invoked = True
        return "should not run"

    tool = McpTool(
        name="t",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )
    session, client = make_session(
        [
            FakeChatStream(chunks=[], tool_calls=[make_tool_call("t", {})]),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=[tool],
        approve_tool=hook,
    )
    await session.send("go")
    assert invoked is False
    second_messages = client.calls[1][0]
    tool_msg = next(m for m in second_messages if m.get("role") == "tool")
    content = json.loads(tool_msg["content"])  # type: ignore[typeddict-item]
    assert "denied" in content["error"]


# Parallel vs sequential execution


@pytest.mark.parametrize(
    "execution, expected_max_concurrent",
    [
        pytest.param("parallel", 2, id="parallel"),
        pytest.param("sequential", 1, id="sequential"),
    ],
)
async def test_execution_policy_controls_concurrency(
    execution: str, expected_max_concurrent: int
) -> None:
    state = {"current": 0, "max": 0}

    async def handler(_args: dict[str, Any]) -> str:
        state["current"] += 1
        state["max"] = max(state["max"], state["current"])
        await asyncio.sleep(0.02)
        state["current"] -= 1
        return "ok"

    tool = McpTool(
        name="t",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )
    session, _ = make_session(
        [
            FakeChatStream(
                chunks=[],
                tool_calls=[
                    make_tool_call("t", {}, call_id="a"),
                    make_tool_call("t", {}, call_id="b"),
                ],
            ),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=[tool],
        execution=execution,  # type: ignore[arg-type]
    )
    await session.send("go")
    assert state["max"] == expected_max_concurrent


# Per-round tool refresh


async def test_per_round_tool_refresh_picks_up_new_tools() -> None:
    async def handler_a(_args: dict[str, Any]) -> str:
        return "a"

    async def handler_b(_args: dict[str, Any]) -> str:
        return "b"

    tool_a = McpTool(
        name="a",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=handler_a,
    )
    tool_b = McpTool(
        name="b",
        description="",
        input_schema={"type": "object", "properties": {}},
        handler=handler_b,
    )

    tools: list[McpTool] = [tool_a]

    # When the model calls `a`, mutate the registry to add `b` before round 2.
    def maybe_add_b(name: str, _args: dict[str, Any]) -> None:
        if name == "a" and tool_b not in tools:
            tools.append(tool_b)

    session, client = make_session(
        [
            FakeChatStream(chunks=[], tool_calls=[make_tool_call("a", {})]),
            FakeChatStream(
                chunks=[], tool_calls=[make_tool_call("b", {}, call_id="c2")]
            ),
            FakeChatStream(chunks=["done"]),
        ],
        mcp_tools=tools,
        on_tool=maybe_add_b,
    )

    await session.send("go")

    # First call should expose only `a`; second call should expose both
    first_tool_names = [t["name"] for t in client.calls[0][1]]
    second_tool_names = [t["name"] for t in client.calls[1][1]]
    assert first_tool_names == ["a"]
    assert sorted(second_tool_names) == ["a", "b"]


# Snapshot and store


def test_conversation_state_snapshot_round_trip() -> None:
    state = ConversationState(system="be helpful")
    state.add_user_message("hi")
    state.add_assistant_message("hello")
    state.accumulate_usage({"total": 12, "input": 7, "output": 5})

    snap = state.snapshot()
    restored = ConversationState.from_snapshot(snap)

    assert restored.system == "be helpful"
    assert restored.messages == state.messages
    assert restored.total_usage == {"total": 12, "input": 7, "output": 5}
    # The system message is not double-inserted
    assert sum(1 for m in restored.messages if m.get("role") == "system") == 1


@pytest.mark.parametrize(
    "source_chunks, source_system, load_system, expected_system",
    [
        pytest.param(
            ["hello"],
            "be helpful",
            None,
            "be helpful",
            id="round_trips_state",
        ),
        pytest.param(
            ["x"],
            "original",
            "overridden — should be ignored",
            "original",
            id="ignores_system_kwarg",
        ),
    ],
)
async def test_load_restores_state_from_store(
    source_chunks: list[str],
    source_system: str,
    load_system: str | None,
    expected_system: str,
) -> None:
    store = FakeStore()
    session, _ = make_session(
        [FakeChatStream(chunks=source_chunks)],
        store=store,
        session_id="abc",
        system=source_system,
    )
    await session.send("hi")
    session.save()

    client2 = FakeClient(responses=[FakeChatStream(chunks=["y"])])
    session2 = AgentSession.load(
        client=cast(LLMClientBase, client2),
        store=store,
        session_id="abc",
        system=load_system,
    )
    assert session2.messages == session.messages
    assert session2.system == expected_system


@pytest.mark.parametrize(
    "session_id_arg, expected_id_check",
    [
        # `session_id=None` → fresh id is generated; just check it's non-empty.
        pytest.param(None, lambda sid: bool(sid), id="none_generates_fresh_id"),
        # `session_id="..."` but missing in store → fresh session, id pinned.
        pytest.param(
            "not-yet-saved",
            lambda sid: sid == "not-yet-saved",
            id="unknown_id_pins_to_caller_value",
        ),
    ],
)
async def test_load_starts_fresh_when_no_snapshot(
    session_id_arg: str | None,
    expected_id_check: Any,
) -> None:
    """`AgentSession.load()` is a get-or-create entry point.

    Either `session_id=None` or a `session_id` that has no snapshot in the
    store should produce an empty session that's still wired up to the
    store, so the first `save()` lands under the right id. The previous
    behaviour required callers to probe the store themselves before
    deciding whether to construct or restore.
    """
    store = FakeStore()
    fake = FakeClient(responses=[FakeChatStream(chunks=["hi"])])
    session = AgentSession.load(
        client=cast(LLMClientBase, fake),
        store=store,
        session_id=session_id_arg,
    )
    assert session.messages == []
    assert expected_id_check(session.session_id)
    assert session.store is store
    await session.send("hi")
    session.save()
    assert session.session_id in store.storage


async def test_load_rejects_both_model_and_client() -> None:
    """Regression: load() silently dropped `client=` when `model=` was also set.

    That broke callers mid-migration who kept their custom client while
    adding model= — their client would be swapped for a fresh `LLMClient`
    with default auth, turning tests into real network calls.
    """
    store = FakeStore()
    fake = FakeClient(responses=[FakeChatStream(chunks=["y"])])
    with pytest.raises(ValueError, match="exactly one of model= or client="):
        AgentSession.load(
            client=cast(LLMClientBase, fake),
            model="gpt-4o",
            store=store,
            session_id="abc",
        )


@dataclass
class _FakeTransport:
    """Minimal McpTransport stand-in for collision/prefix tests.

    Holds a static list of tools and exposes the `name_prefix` /
    `auto_prefix` / `is_open` / `tools` surface that AgentSession
    consults. Doesn't open/close anything.
    """

    _tools: list[McpTool]
    name_prefix: str | None = None
    auto_prefix_value: str = "transport"
    label_value: str = "fake-transport"

    @property
    def tools(self) -> list[McpTool]:
        return self._tools

    @property
    def is_open(self) -> bool:
        return True

    @property
    def auto_prefix(self) -> str:
        return self.auto_prefix_value

    @property
    def label(self) -> str:
        return self.label_value

    async def __aenter__(self) -> "_FakeTransport":
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def ping(self) -> None:
        pass


def _make_local_tool(name: str, payload: str) -> McpTool:
    async def handler(_args: dict[str, Any]) -> str:
        return payload

    return McpTool(
        name=name,
        description=name,
        input_schema={"type": "object", "properties": {}},
        handler=handler,
    )


@pytest.mark.parametrize(
    "scenario, expected_names, expected_error",
    [
        pytest.param(
            "duplicate_local_tools",
            None,
            "Duplicate tool name 'search'",
            id="duplicate_local_tools_raise",
        ),
        pytest.param(
            "auto_prefix_transports",
            {"src_a__search", "src_a__filter", "src_b__search"},
            None,
            id="collision_auto_prefixes_transports",
        ),
        pytest.param(
            "skip_explicit_prefix",
            {"weather__search", "search"},
            None,
            id="collision_skips_transport_with_explicit_prefix",
        ),
        pytest.param(
            "duplicate_explicit_prefixes",
            None,
            "Duplicate tool name 'weather__search'",
            id="collision_with_two_explicit_prefixes_colliding_raises",
        ),
    ],
)
async def test_tool_name_collision_handling(
    scenario: str,
    expected_names: set[str] | None,
    expected_error: str | None,
) -> None:
    """Covers duplicate-name behaviour for local tools and transports."""
    if scenario == "duplicate_local_tools":
        mcp_tools: Sequence[McpTool] = [
            _make_local_tool("search", "a"),
            _make_local_tool("search", "b"),
        ]
    elif scenario == "auto_prefix_transports":
        transport_a = _FakeTransport(
            _tools=[
                _make_local_tool("search", "a-search"),
                _make_local_tool("filter", "a-filter"),
            ],
            auto_prefix_value="src_a",
        )
        transport_b = _FakeTransport(
            _tools=[_make_local_tool("search", "b-search")],
            auto_prefix_value="src_b",
        )
        mcp_tools = cast(Sequence[McpTool], [transport_a, transport_b])
    elif scenario == "skip_explicit_prefix":
        transport_a = _FakeTransport(
            _tools=[_make_local_tool("weather__search", "a")],
            name_prefix="weather",
        )
        transport_b = _FakeTransport(
            _tools=[_make_local_tool("search", "b")],
            auto_prefix_value="src_b",
        )
        mcp_tools = cast(Sequence[McpTool], [transport_a, transport_b])
    else:
        transport_a = _FakeTransport(
            _tools=[_make_local_tool("weather__search", "a")],
            name_prefix="weather",
        )
        transport_b = _FakeTransport(
            _tools=[_make_local_tool("weather__search", "b")],
            name_prefix="weather",
        )
        mcp_tools = cast(Sequence[McpTool], [transport_a, transport_b])

    session, client = make_session(
        [FakeChatStream(chunks=["done" if expected_error is None else "unused"])],
        mcp_tools=mcp_tools,
    )

    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            await session.send("hi")
        return

    await session.send("hi")
    advertised = {t["name"] for t in client.calls[0][1]}
    assert advertised == expected_names


# `_extract_text` regression tests for tool result normalization


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param("plain text", "plain text", id="plain-string"),
        pytest.param(
            {"content": [{"type": "text", "text": "only block"}]},
            "only block",
            id="single-text-block",
        ),
        pytest.param(
            {
                "content": [
                    {"type": "text", "text": "summary"},
                    {"type": "text", "text": "details"},
                ]
            },
            "summary\ndetails",
            id="multiple-text-blocks-concatenated",
        ),
        pytest.param(
            {
                "content": [
                    {"type": "image", "data": "base64..."},
                    {"type": "text", "text": "caption"},
                    {"type": "resource", "uri": "file://x"},
                    {"type": "text", "text": "footer"},
                ]
            },
            json.dumps(
                {
                    "content": [
                        {"type": "image", "data": "base64..."},
                        {"type": "text", "text": "caption"},
                        {"type": "resource", "uri": "file://x"},
                        {"type": "text", "text": "footer"},
                    ]
                }
            ),
            id="non-text-blocks-roundtrip-as-json",
        ),
        pytest.param(
            {"content": []},
            '{"content": []}',
            id="no-text-blocks-fallback-to-json",
        ),
        pytest.param(
            {"foo": "bar"},
            '{"foo": "bar"}',
            id="non-mcp-dict-fallback-to-json",
        ),
        pytest.param(
            {
                "content": [{"type": "text", "text": "summary"}],
                "structuredContent": {"answer": 42},
            },
            json.dumps(
                {
                    "content": [{"type": "text", "text": "summary"}],
                    "structuredContent": {"answer": 42},
                }
            ),
            id="text-with-structured-content-roundtrips-as-json",
        ),
        pytest.param(
            {
                "content": [{"type": "text", "text": "boom"}],
                "isError": True,
            },
            json.dumps(
                {
                    "content": [{"type": "text", "text": "boom"}],
                    "isError": True,
                }
            ),
            id="is-error-flag-preserved",
        ),
        pytest.param(
            {"structuredContent": {"answer": 42}},
            json.dumps({"structuredContent": {"answer": 42}}),
            id="structured-content-only",
        ),
    ],
)
def test_extract_text(result: Any, expected: str) -> None:
    """Regression: `_extract_text` used to return only the first text
    block from an MCP tool result. The first round of fixes started
    concatenating every text block — but still silently dropped
    `structuredContent`, `isError`, and non-text content blocks
    (image/resource), which made JSON-only MCP tools unusable
    end-to-end. The agent now JSON-encodes any dict that carries
    fields beyond pure text blocks so the model sees the full payload.
    """
    assert _extract_text(result) == expected


# Startup-failure rollback tests


@dataclass
class _FailingTransport:
    """Fake McpTransport that raises during `__aenter__` (after the first
    successful entry of an earlier resource).
    """

    fail_message: str = "boom: server unavailable"
    aenter_count: int = 0
    aexit_count: int = 0
    _is_open: bool = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def tools(self) -> list[McpTool]:
        return []

    async def __aenter__(self) -> "_FailingTransport":
        self.aenter_count += 1
        raise RuntimeError(self.fail_message)

    async def __aexit__(self, *_: object) -> None:
        self.aexit_count += 1
        self._is_open = False

    async def ping(self) -> None:
        pass


async def test_aenter_unwinds_on_partial_failure() -> None:
    """Regression: if `client` opened successfully and then a transport
    fails to enter, `AgentSession.__aenter__` returned with the exit
    stack still owning the live client (and earlier transports), but
    `__aexit__` would never run because the constructor was raising.
    The leak left HTTP sessions and subprocesses alive across retries.
    """
    fake = FakeClient(responses=[FakeChatStream(chunks=["ignored"])])
    failing = _FailingTransport()

    with pytest.raises(RuntimeError, match="boom: server unavailable"):
        async with AgentSession(
            client=cast(LLMClientBase, fake),
            mcp_tools=cast(Sequence[McpTool], [failing]),
        ):
            pytest.fail("AgentSession.__aenter__ should have raised")

    # The client we successfully opened must have been closed by the
    # rollback, not left dangling.
    assert fake.aenter_count == 1
    assert fake.aexit_count == 1
    assert fake.is_open is False
    # The failing transport's `__aenter__` was attempted exactly once.
    assert failing.aenter_count == 1


# on_mcp_connect


@dataclass
class _PingFailTransport(_FakeTransport):
    """Transport whose ping() raises."""

    async def ping(self) -> None:
        raise RuntimeError("ping failed")


@pytest.mark.parametrize(
    "transports, expected_count",
    [
        pytest.param(
            lambda: [
                _FakeTransport(_tools=[]),
                _FakeTransport(_tools=[]),
            ],
            2,
            id="fires-per-transport",
        ),
        pytest.param(
            lambda: [],
            0,
            id="none-is-silent",
        ),
    ],
)
async def test_on_mcp_connect(transports, expected_count) -> None:
    connected: list[object] = []
    fake = FakeClient(responses=[FakeChatStream(chunks=["hi"])])
    async with AgentSession(
        client=cast(LLMClientBase, fake),
        mcp_tools=cast(Sequence[McpTool], transports()),
        on_mcp_connect=connected.append,
    ):
        pass
    assert len(connected) == expected_count


async def test_on_mcp_connect_ping_failure_unwinds() -> None:
    """If ping() fails, __aenter__ raises and the exit stack unwinds."""
    fake = FakeClient(responses=[FakeChatStream(chunks=["hi"])])
    failing = _PingFailTransport(_tools=[])
    with pytest.raises(RuntimeError, match="ping failed"):
        async with AgentSession(
            client=cast(LLMClientBase, fake),
            mcp_tools=cast(Sequence[McpTool], [failing]),
        ):
            pytest.fail("should have raised")
    assert fake.aexit_count == 1
