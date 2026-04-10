import asyncio
import contextlib
import inspect
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self

from ._base import LLMClientBase
from ._json import dumps as _json_dumps, loads as _json_loads
from .client import LLMClient
from .conversation import (
    AssistantToolMessage,
    ChatMessage,
    ConversationSnapshot,
    ConversationState,
    ToolResultMessage,
)
from .logs import log
from .mcp import McpTool, McpTransport
from .models import ToolCall, ToolDefinition, UsageToken

__all__ = ("AgentSession", "ConversationStore")

type ToolErrorHandler = Callable[[McpTool, dict[str, Any], Exception], str]
type ApprovalHook = Callable[[McpTool, dict[str, Any]], bool | Awaitable[bool]]


class ConversationStore(Protocol):
    """Persistence backend for conversation snapshots.

    Implementations should round-trip a `ConversationSnapshot` keyed by an
    opaque session id.
    """

    def save(self, session_id: str, snapshot: ConversationSnapshot) -> None: ...

    def load(self, session_id: str) -> ConversationSnapshot: ...


def _extract_text(result: Any) -> str:
    """Reduce a tool result to a string suitable for ToolResultMessage.content.

    Plain strings pass through unchanged::

        "Hello world"  →  "Hello world"

    A dict whose only field is a ``content`` array of pure text blocks
    collapses to newline-joined text::

        {"content": [
            {"type": "text", "text": "File contents here"},
            {"type": "text", "text": "More output"}
        ]}
        →  "File contents here\nMore output"

    Anything richer (``structuredContent``, ``isError``, non-text blocks)
    round-trips as JSON so nothing is dropped::

        {"content": [{"type": "text", "text": "summary"}],
         "structuredContent": {"rows": [{"id": 1, "name": "foo"}]},
         "isError": false}
        →  '{"content": [...], "structuredContent": {...}, "isError": false}'
    """
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return _json_dumps(result)

    content = result.get("content") or []
    if set(result.keys()) <= {"content"}:
        texts: list[str] = []
        for b in content:
            if not isinstance(b, dict) or b.get("type") != "text":
                break
            if t := b.get("text"):
                texts.append(t)
        else:
            if texts:
                return "\n".join(texts)
    # Anything richer than pure text blocks (or pure text alongside
    # structured/error fields) round-trips as JSON so nothing is dropped.
    return _json_dumps(result)


@dataclass
class AgentSession:
    """Multi-turn conversation runner with streaming and tool dispatch.

    Wraps a `ConversationState` with the loop that calls the LLM, dispatches
    any tool calls in the response, feeds the results back, and repeats until
    the model produces a plain text answer (or `max_tool_rounds` LLM calls
    have been made).

    The tool list is read from `mcp_tools` at the top of every round, so
    consumers that mutate the sequence (e.g. when an MCP server emits
    `notifications/tools/list_changed`, which causes
    `McpStreamable.tools` / `McpStdio.tools` to refresh in place) get the
    updated set on the next iteration without restarting the session.
    """

    client: LLMClientBase
    system: str | None = None
    mcp_tools: Sequence[McpTool | McpTransport] = field(default_factory=list)
    max_tool_rounds: int | None = 5
    """Max LLM round-trips per ``send()`` call. None disables the limit."""
    max_tool_result_chars: int | None = 8_000
    """Truncate tool results sent back to the model. None disables truncation."""
    execution: Literal["sequential", "parallel"] = "sequential"
    """Run tool calls sequentially or in parallel within each round."""
    on_tool: Callable[[str, dict[str, Any]], None] | None = None
    """Hook notified before each tool dispatch, for logging or side-effects."""
    on_tool_error: ToolErrorHandler | None = None
    """Custom error handler, replaces the default JSON-serialised error message."""
    approve_tool: ApprovalHook | None = None
    """Gate that can deny individual tool calls before they execute."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    store: ConversationStore | None = None
    _state: ConversationState = field(init=False)
    _exit_stack: contextlib.AsyncExitStack = field(init=False)
    _dispatch_cache: tuple[list[ToolDefinition], dict[str, McpTool]] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _dispatch_fingerprint: tuple[int, ...] | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.max_tool_rounds is not None and self.max_tool_rounds < 1:
            raise ValueError(
                f"max_tool_rounds must be >= 1 or None, got {self.max_tool_rounds}"
            )
        self._state = ConversationState(system=self.system)
        self._exit_stack = contextlib.AsyncExitStack()

    @property
    def messages(self) -> list[ChatMessage]:
        return self._state.messages

    @property
    def last_usage(self) -> UsageToken | None:
        return self._state.last_usage

    @property
    def total_usage(self) -> UsageToken:
        return self._state.total_usage

    def add_user_message(self, content: str) -> None:
        self._state.add_user_message(content)

    def clear(self) -> None:
        self._state.clear()

    async def __aenter__(self) -> Self:
        """Enter the session, opening any resources the caller hasn't already.

        Resources (`client` and each `McpTransport` in `mcp_tools`) that are
        already open — detected via `is_open` — are left alone: the caller
        still owns their lifecycle. Only resources we open here are
        registered with the exit stack for cleanup on `__aexit__`.

        If any resource fails to enter after others have already been
        opened, the exit stack is unwound before the exception propagates,
        so a partial startup cannot leak live HTTP sessions or MCP
        subprocesses. Without this rollback, `__aexit__` would never run
        (since `__aenter__` raised) and previously-entered resources
        would stay alive until the event loop shuts down.
        """
        await self._exit_stack.__aenter__()
        try:
            await self._exit_stack.enter_async_context(self.client)
            for item in self.mcp_tools:
                if isinstance(item, McpTool):
                    continue
                if not item.is_open:
                    await self._exit_stack.enter_async_context(item)
        except BaseException:
            await self._exit_stack.aclose()
            raise
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._exit_stack.__aexit__(*args)

    def save(self) -> None:
        """Persist the current state via the configured store, if any."""
        if self.store is None:
            return
        self.store.save(self.session_id, self._state.snapshot())

    @classmethod
    def load(
        cls,
        *,
        store: ConversationStore,
        session_id: str | None = None,
        model: str | None = None,
        client: LLMClientBase | None = None,
        system: str | None = None,
        mcp_tools: Sequence[McpTool | McpTransport] = (),
        max_tool_rounds: int | None = 5,
        max_tool_result_chars: int | None = 8_000,
        execution: Literal["sequential", "parallel"] = "sequential",
        on_tool: Callable[[str, dict[str, Any]], None] | None = None,
        on_tool_error: ToolErrorHandler | None = None,
        approve_tool: ApprovalHook | None = None,
    ) -> Self:
        """Construct an AgentSession, optionally restoring state from `store`.

        If `session_id` is provided and the store has a matching snapshot,
        the conversation history, total usage, and system prompt are
        restored from it. If the snapshot is missing, the session is built fresh but
        pinned to the caller-supplied `session_id`, so subsequent `save()`
        calls land under that id.

        If `session_id` is `None`, a fresh id is generated and the session
        starts empty; future `save()` calls still go to the configured
        store.

        Pass `model` to have an `LLMClient` created automatically, or
        `client` for a pre-configured or fake client. Exactly one must
        be provided.

        When restoring, the `system` prompt is taken from the persisted
        snapshot; the `system` parameter is ignored.
        """
        if model is not None and client is not None:
            raise ValueError("Provide exactly one of model= or client=, not both")
        if model is not None:
            _client: LLMClientBase = LLMClient(model=model)
        elif client is not None:
            _client = client
        else:
            raise ValueError("Either model= or client= must be provided")

        snapshot: ConversationSnapshot | None = None
        if session_id is not None:
            try:
                snapshot = store.load(session_id)
            except LookupError:
                pass

        instance = cls(
            client=_client,
            store=store,
            system=snapshot.get("system") if snapshot else system,
            **({} if session_id is None else {"session_id": session_id}),
            mcp_tools=mcp_tools,
            max_tool_rounds=max_tool_rounds,
            max_tool_result_chars=max_tool_result_chars,
            execution=execution,
            on_tool=on_tool,
            on_tool_error=on_tool_error,
            approve_tool=approve_tool,
        )
        if snapshot:
            instance._state = ConversationState.from_snapshot(snapshot)
        return instance

    def _context_messages(self) -> list[ChatMessage]:
        """Return messages with tool results truncated to `max_tool_result_chars`.

        Truncation only affects the copy sent to the LLM; the full content is
        retained in `self.messages`. Pass `max_tool_result_chars=None` to
        disable truncation entirely.
        """
        limit = self.max_tool_result_chars
        if limit is None:
            return list(self._state.messages)
        out: list[ChatMessage] = []
        for msg in self._state.messages:
            if msg["role"] == "tool":
                content = msg["content"]
                if len(content) > limit:
                    msg = ToolResultMessage(
                        role="tool",
                        tool_call_id=msg["tool_call_id"],
                        name=msg["name"],
                        content=content[:limit] + "\n[truncated]",
                    )
            out.append(msg)
        return out

    def _build_round_dispatch(
        self,
    ) -> tuple[list[ToolDefinition], dict[str, McpTool]]:
        """Build the per-round tool registry, auto-prefixing on collision.

        First pass: collect every tool from every item in `mcp_tools`,
        tracking which transport (if any) it came from. If two tools
        end up with the same public name, the colliding **transports**
        get their `auto_prefix` applied to *all* their tools (consistent
        naming inside one server) — but only if the transport doesn't
        already have an explicit `name_prefix` set, since explicit
        wins. Local `McpTool` instances stay bare regardless.

        If even auto-prefixing leaves duplicates (two transports with
        the same auto-derived prefix, two local tools with the same
        name, or an explicit prefix that still collides), we raise so
        the operator fixes the configuration at the source.
        """
        # (tool, source_transport_or_None) for everything in mcp_tools.
        items: list[tuple[McpTool, McpTransport | None]] = []
        tool_ids: list[int] = []
        for item in self.mcp_tools:
            if isinstance(item, McpTool):
                items.append((item, None))
                tool_ids.append(id(item))
            else:
                for t in item.tools:
                    items.append((t, item))
                    tool_ids.append(id(t))

        fingerprint = tuple(tool_ids)
        if (
            fingerprint == self._dispatch_fingerprint
            and self._dispatch_cache is not None
        ):
            return self._dispatch_cache

        # First pass: count names to find collisions.
        name_counts: dict[str, int] = {}
        for tool, _ in items:
            name_counts[tool.name] = name_counts.get(tool.name, 0) + 1

        # Determine which transports need auto-prefixing. Only those
        # contributing a colliding name AND with no explicit prefix.
        needs_auto: set[int] = set()
        for tool, src in items:
            if (
                name_counts[tool.name] > 1
                and src is not None
                and src.name_prefix is None
            ):
                needs_auto.add(id(src))

        if needs_auto:
            log.warning(
                "MCP tool name collision detected; auto-prefixing %d transport(s) "
                "with their derived prefix. Set `name_prefix=` on each "
                "transport to control the namespace explicitly.",
                len(needs_auto),
            )

        tool_defs: list[ToolDefinition] = []
        dispatch: dict[str, McpTool] = {}
        for tool, src in items:
            if src is not None and id(src) in needs_auto:
                prefix = src.auto_prefix
                public = McpTool(
                    name=f"{prefix}__{tool.name}",
                    description=tool.description,
                    input_schema=tool.input_schema,
                    handler=tool.handler,
                )
            else:
                public = tool
            if public.name in dispatch:
                raise ValueError(
                    f"Duplicate tool name {public.name!r} in mcp_tools after "
                    "auto-prefixing. Set an explicit `name_prefix=` on the "
                    "conflicting transport(s), or rename a local McpTool, "
                    "so every tool has a unique public name."
                )
            tool_defs.append(public.to_tool_def())
            dispatch[public.name] = public
        self._dispatch_fingerprint = fingerprint
        self._dispatch_cache = (tool_defs, dispatch)
        return tool_defs, dispatch

    async def _approve(self, tool: McpTool, args: dict[str, Any]) -> bool:
        if self.approve_tool is None:
            return True
        decision = self.approve_tool(tool, args)
        if inspect.isawaitable(decision):
            decision = await decision
        return bool(decision)

    async def _dispatch_one(
        self,
        tc: ToolCall,
        tool: McpTool | None,
        args: dict[str, Any],
        approved: bool,
    ) -> str:
        name = tc["function"]["name"]
        if tool is None:
            return _json_dumps({"error": f"Unknown tool: {name}"})
        if not approved:
            return _json_dumps({"error": f"Tool call denied by approval hook: {name}"})
        try:
            result: Any = tool.handler(args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            log.warning("Tool %r raised: %s", name, exc, exc_info=True)
            if self.on_tool_error is not None:
                return self.on_tool_error(tool, args, exc)
            return _json_dumps({"error": str(exc)})
        return _extract_text(result)

    async def _run_tool_calls(
        self,
        tool_calls: Sequence[ToolCall],
        dispatch: dict[str, McpTool],
    ) -> None:
        """Pre-flight (`on_tool` + `approve_tool`) sequentially, then dispatch.

        Dispatch runs sequentially or via `asyncio.gather` depending on
        `self.execution`. Results are appended to state in original call
        order regardless of execution policy.
        """
        plan: list[tuple[ToolCall, McpTool | None, dict[str, Any], bool]] = []
        for tc in tool_calls:
            name = tc["function"]["name"]
            try:
                args = _json_loads(tc["function"]["arguments"])
            except ValueError as exc:
                log.warning("Bad tool args for %r: %s", name, exc)
                args = {}
            if self.on_tool is not None:
                self.on_tool(name, args)
            tool = dispatch.get(name)
            approved = await self._approve(tool, args) if tool is not None else False
            plan.append((tc, tool, args, approved))

        if self.execution == "parallel":
            results = await asyncio.gather(
                *(self._dispatch_one(tc, tool, args, ok) for tc, tool, args, ok in plan)
            )
        else:
            results = [
                await self._dispatch_one(tc, tool, args, ok)
                for tc, tool, args, ok in plan
            ]

        for (tc, _tool, _args, _ok), result_str in zip(plan, results, strict=True):
            self._state.add_tool_result(tc["id"], tc["function"]["name"], result_str)

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """Send a message and stream the response, dispatching tool calls.

        Yields text chunks as the model produces them across all rounds. Tool
        calls are silent on the stream — observe them via `on_tool`. The
        iterator finishes once the model returns a plain text response with
        no further tool calls, or once `max_tool_rounds` LLM calls have been
        made (in which case a final limit-reached message is yielded).
        """
        self._state.add_user_message(user_input)
        calls_remaining = self.max_tool_rounds

        while calls_remaining is None or calls_remaining > 0:
            if calls_remaining is not None:
                calls_remaining -= 1

            # Read mcp_tools at the top of every round — picks up tools/list_changed updates.
            tool_defs, dispatch = self._build_round_dispatch()
            chunks: list[str] = []
            chat_stream = self.client.stream_chat(
                self._context_messages(),
                tools=tool_defs or None,
            )

            async for text in chat_stream:
                chunks.append(text)
                yield text

            if chat_stream.usage:
                self._state.accumulate_usage(chat_stream.usage)

            if not chat_stream.tool_calls:
                text = "".join(chunks)
                if not text:
                    text = "(no response)"
                    yield text
                self._state.add_assistant_message(text)
                return

            self._state.messages.append(
                AssistantToolMessage(
                    role="assistant",
                    content="".join(chunks) or None,
                    tool_calls=chat_stream.tool_calls,
                )
            )
            await self._run_tool_calls(chat_stream.tool_calls, dispatch)

        msg = (
            f"(reached tool call limit of {self.max_tool_rounds} rounds "
            "without a final answer)"
        )
        log.warning(msg)
        yield msg

    async def send(self, user_input: str) -> str:
        """Send a message and return the complete response (non-streaming)."""
        chunks: list[str] = []
        async for chunk in self.stream(user_input):
            chunks.append(chunk)
        return "".join(chunks)
