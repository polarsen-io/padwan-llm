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
from padwan_llm.agent import ToolErrorHandler


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

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolDefinition] | None = None,
    ) -> ChatStream:
        self.calls.append((list(messages), list(tools or [])))
        return self.responses.pop(0)


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


# Basic loop behaviour


async def test_plain_text_response_finishes_in_one_round() -> None:
    session, client = make_session(
        [FakeChatStream(chunks=["Hello", " world"])],
    )
    out = await session.send("hi")
    assert out == "Hello world"
    assert len(client.calls) == 1
    assert session.messages[-1] == {"role": "assistant", "content": "Hello world"}


async def test_empty_response_yields_no_response_marker() -> None:
    session, _ = make_session([FakeChatStream(chunks=[])])
    out = await session.send("hi")
    assert out == "(no response)"
    assert session.messages[-1] == {"role": "assistant", "content": "(no response)"}


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


async def test_long_tool_result_truncated_in_context_only() -> None:
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
        max_tool_result_chars=100,
    )
    await session.send("go")

    # Stored history keeps the full result
    stored = next(m for m in session.messages if m.get("role") == "tool")
    assert len(stored["content"]) == 50_000  # type: ignore[typeddict-item]

    # The context sent to the second LLM call is truncated
    second_messages = client.calls[1][0]
    sent = next(m for m in second_messages if m.get("role") == "tool")
    assert len(sent["content"]) <= 200  # type: ignore[typeddict-item]
    assert sent["content"].endswith("[truncated]")  # type: ignore[typeddict-item]


async def test_max_tool_result_chars_none_disables_truncation() -> None:
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
        max_tool_result_chars=None,
    )
    await session.send("go")

    second_messages = client.calls[1][0]
    sent = next(m for m in second_messages if m.get("role") == "tool")
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


async def test_parallel_execution_runs_handlers_concurrently() -> None:
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
        execution="parallel",
    )
    await session.send("go")
    assert state["max"] == 2


async def test_sequential_execution_runs_handlers_one_at_a_time() -> None:
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
    )
    await session.send("go")
    assert state["max"] == 1


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


async def test_save_and_load_via_store_round_trips_state() -> None:
    storage: dict[str, ConversationSnapshot] = {}

    class FakeStore:
        def save(self, session_id: str, snapshot: ConversationSnapshot) -> None:
            storage[session_id] = snapshot

        def load(self, session_id: str) -> ConversationSnapshot:
            return storage[session_id]

    store = FakeStore()
    session, _ = make_session(
        [FakeChatStream(chunks=["hello"])],
        store=store,
        session_id="abc",
        system="be helpful",
    )
    await session.send("hi")
    session.save()

    # Restore via the classmethod constructor — single call.
    client2 = FakeClient(responses=[FakeChatStream(chunks=["world"])])
    session2 = AgentSession.load(
        client=cast(LLMClientBase, client2),
        store=store,
        session_id="abc",
    )
    assert session2.messages == session.messages
    assert session2.system == "be helpful"


async def test_load_classmethod_ignores_system_kwarg() -> None:
    """The system prompt is taken from the snapshot, not the kwargs."""
    storage: dict[str, ConversationSnapshot] = {}

    class FakeStore:
        def save(self, session_id: str, snapshot: ConversationSnapshot) -> None:
            storage[session_id] = snapshot

        def load(self, session_id: str) -> ConversationSnapshot:
            return storage[session_id]

    store = FakeStore()
    session, _ = make_session(
        [FakeChatStream(chunks=["x"])],
        store=store,
        session_id="abc",
        system="original",
    )
    await session.send("hi")
    session.save()

    client2 = FakeClient(responses=[FakeChatStream(chunks=["y"])])
    session2 = AgentSession.load(
        client=cast(LLMClientBase, client2),
        store=store,
        session_id="abc",
        system="overridden — should be ignored",
    )
    assert session2.system == "original"
