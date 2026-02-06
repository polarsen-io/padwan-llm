from __future__ import annotations

from typing import TYPE_CHECKING

from piou import Option, CommandGroup
from piou.tui import PromptStyle, TuiContext, TuiOption

from rich.console import Console

from .. import LLMClient

from .conversation import Conversation, Message
from .widgets import StreamingMessage, UserMessage

if TYPE_CHECKING:
    from .. import LLMClientBase

console = Console()

chat_group = CommandGroup(name="chat", help="Chat with an LLM")

# Persistent conversation sessions keyed by model name
_sessions: dict[str, tuple[LLMClientBase, Conversation]] = {}


def _get_or_create_session(
    model: str,
    system: str | None = None,
) -> tuple[LLMClientBase, Conversation]:
    """Get existing session or create a new one for the given model."""
    if model in _sessions:
        client, conv = _sessions[model]
        # Update system prompt if provided and different
        if system and conv.system != system:
            conv.system = system
            # Rebuild messages with new system prompt
            old_messages = [m for m in conv.messages if m["role"] != "system"]
            conv.messages = []
            if system:
                conv.messages.append(Message(role="system", content=system))
            conv.messages.extend(old_messages)
        return client, conv

    client = LLMClient(model=model)
    conv = Conversation(client, system=system)
    _sessions[model] = (client, conv)
    return client, conv


CHAT_PROMPT = PromptStyle(text="You: ", css_class="chat-mode")


def _format_tokens(conv: Conversation, ctx: TuiContext | None = None) -> str:
    """Format token usage for display."""
    if not conv.last_usage:
        return ""
    last = conv.last_usage
    total = conv.total_usage
    parts = [f"in: {last['input']}", f"out: {last['output']}"]
    if "cached" in last:
        parts.append(f"cached: {last['cached']}")
    parts.append(f"| session: {total['total']}")
    if ctx and ctx.pending_count > 0:
        parts.append(f"| {ctx.pending_count} queued")
    return " ".join(parts)


@chat_group.command("send", help="Send a message to the LLM")
async def chat_send_fn(
    message: str = Option(..., help="Message to send"),
    model: str = Option("gpt-4o-mini", "-m", "--model", help="Model to use"),
    ctx: TuiContext = TuiOption(),
) -> None:
    """Start a conversation. Use Ctrl+C to exit."""
    client, conv = _get_or_create_session(model)

    async with client:
        user_input: str | None = message
        original_style = ctx.set_prompt_style(CHAT_PROMPT)
        ctx.set_hint("Chat mode - press Ctrl+C to exit")
        ctx.set_rule_above(add_class="chat-mode")
        ctx.set_rule_below(add_class="chat-mode")

        try:
            while user_input:
                if ctx.is_tui:
                    ctx.mount_widget(UserMessage(user_input))
                    widget = StreamingMessage()
                    ctx.mount_widget(widget)
                    ctx.set_hint("Responding...")
                    ctx.set_silent_queue(True)
                    async for chunk in conv.stream(user_input):
                        widget.append(chunk)
                        if (n := ctx.pending_count) > 0:
                            ctx.set_hint(f"Responding... ({n} queued)")
                    ctx.set_silent_queue(False)
                    ctx.set_hint("Chat mode - press Ctrl+C to exit")
                    ctx.set_status_above(_format_tokens(conv, ctx) or None)
                    user_input = await ctx.prompt()
                else:
                    async for chunk in conv.stream(user_input):
                        console.print(chunk, end="")
                    console.print()
                    if tokens := _format_tokens(conv):
                        console.print(f"[dim]{tokens}[/dim]")
                    break
        finally:
            ctx.set_silent_queue(False)
            ctx.clear_queue()  # Discard any messages typed during streaming
            ctx.set_status_above(None)
            ctx.set_hint(None)
            ctx.set_rule_above(remove_class="chat-mode")
            ctx.set_rule_below(remove_class="chat-mode")
            if original_style:
                ctx.set_prompt_style(original_style)


@chat_group.command("clear", help="Clear conversation history")
async def chat_clear_fn(
    model: str | None = Option(
        None, "-m", "--model", help="Model session to clear (all if omitted)"
    ),
    ctx: TuiContext = TuiOption(),
) -> None:
    """Clear conversation history for a model or all models."""
    if model:
        if model in _sessions:
            _sessions[model][1].clear()
            ctx.notify(f"Cleared history for {model}", title="Chat")
            print(f"Cleared conversation history for {model}")
        else:
            print(f"No active session for {model}")
    else:
        for _, conv in _sessions.values():
            conv.clear()
        ctx.notify("Cleared all chat history", title="Chat")
        print("Cleared all conversation history")
