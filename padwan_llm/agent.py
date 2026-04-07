import asyncio
import inspect
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Self

from ._base import LLMClientBase
from .conversation import (
    AssistantToolMessage,
    ChatMessage,
    ConversationSnapshot,
    ConversationState,
    ToolResultMessage,
)
from .logs import log
from .mcp import McpTool
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

    Recognises the MCP wire format (`{"content": [{"type": "text", "text":
    ...}]}`), passes plain strings through unchanged, and falls back to
    JSON-encoding any other value.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text") or ""
        return json.dumps(result)
    return json.dumps(result)


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
    mcp_tools: Sequence[McpTool] = field(default_factory=list)
    max_tool_rounds: int | None = 30
    max_tool_result_chars: int | None = 8_000
    execution: Literal["sequential", "parallel"] = "sequential"
    on_tool: Callable[[str, dict[str, Any]], None] | None = None
    on_tool_error: ToolErrorHandler | None = None
    approve_tool: ApprovalHook | None = None
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    store: ConversationStore | None = None
    _state: ConversationState = field(init=False)

    def __post_init__(self) -> None:
        if self.max_tool_rounds is not None and self.max_tool_rounds < 1:
            raise ValueError(
                f"max_tool_rounds must be >= 1 or None, got {self.max_tool_rounds}"
            )
        self._state = ConversationState(system=self.system)

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

    def save(self) -> None:
        """Persist the current state via the configured store, if any."""
        if self.store is None:
            return
        self.store.save(self.session_id, self._state.snapshot())

    @classmethod
    def load(
        cls,
        *,
        client: LLMClientBase,
        store: ConversationStore,
        session_id: str,
        **kwargs: Any,
    ) -> Self:
        """Construct an AgentSession with state restored from `store`.

        Convenience constructor that replaces the two-step
        `s = AgentSession(...); s.load()` pattern with a single call. The
        `system` prompt is taken from the persisted snapshot, so any
        `system` value passed in `kwargs` is silently ignored.
        """
        snapshot = store.load(session_id)
        kwargs.pop("system", None)
        instance = cls(
            client=client,
            store=store,
            session_id=session_id,
            system=snapshot.get("system"),
            **kwargs,
        )
        instance._state = ConversationState.from_snapshot(snapshot)
        return instance

    def _build_dispatch(self) -> tuple[list[ToolDefinition], dict[str, McpTool]]:
        """Snapshot `mcp_tools` for the current round into definitions + name lookup."""
        tools = list(self.mcp_tools)
        return [t.to_tool_def() for t in tools], {t.name: t for t in tools}

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
            if msg.get("role") == "tool":
                content = msg.get("content") or ""
                if isinstance(content, str) and len(content) > limit:
                    msg = ToolResultMessage(
                        role="tool",
                        tool_call_id=msg["tool_call_id"],  # type: ignore[typeddict-item]
                        name=msg["name"],  # type: ignore[typeddict-item]
                        content=content[:limit] + "\n[truncated]",
                    )
            out.append(msg)
        return out

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
            return json.dumps({"error": f"Unknown tool: {name}"})
        if not approved:
            return json.dumps({"error": f"Tool call denied by approval hook: {name}"})
        try:
            result: Any = tool.handler(args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            log.warning("Tool %r raised: %s", name, exc, exc_info=True)
            if self.on_tool_error is not None:
                return self.on_tool_error(tool, args, exc)
            return json.dumps({"error": str(exc)})
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
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError as exc:
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

            tool_defs, dispatch = self._build_dispatch()
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
