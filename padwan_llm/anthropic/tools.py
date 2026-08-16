from __future__ import annotations

from collections.abc import Sequence

from .._json import dumps as _json_dumps
from .._json import loads as _json_loads
from ..conversation import AssistantToolMessage, ToolResultMessage
from ..models import ToolCall, ToolCallFunction, ToolDefinition
from .models import AnthropicContentBlock, AnthropicMessage, AnthropicTool

__all__ = ("AnthropicToolMixin",)


class AnthropicToolMixin:
    """Mixin providing tool-calling support for the Anthropic client and stream."""

    @staticmethod
    def _tools_to_anthropic(tools: Sequence[ToolDefinition]) -> list[AnthropicTool]:
        """Convert provider-agnostic ToolDefinitions to Anthropic tool format."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    @staticmethod
    def _extract_anthropic_tool_calls(
        blocks: Sequence[AnthropicContentBlock],
    ) -> list[ToolCall]:
        """Extract ToolCall objects from response content blocks of type tool_use."""
        return [
            ToolCall(
                id=block.get("id", ""),
                type="function",
                function=ToolCallFunction(
                    name=block.get("name", ""),
                    arguments=_json_dumps(block.get("input", {})),
                ),
            )
            for block in blocks
            if block.get("type") == "tool_use"
        ]

    @staticmethod
    def _convert_tool_result(msg: ToolResultMessage) -> AnthropicMessage:
        """Convert a ToolResultMessage to a user message carrying a tool_result block.

        Consecutive same-role messages are combined into a single turn by the
        API, so each result can be its own message.
        """
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": msg["content"],
                }
            ],
        }

    @staticmethod
    def _convert_assistant_tool_message(msg: AssistantToolMessage) -> AnthropicMessage:
        """Convert an AssistantToolMessage to an assistant message with tool_use blocks."""
        blocks: list[AnthropicContentBlock] = []
        if msg["content"]:
            blocks.append({"type": "text", "text": msg["content"]})
        for tc in msg["tool_calls"]:
            try:
                parsed_args = _json_loads(tc["function"]["arguments"])
            except (ValueError, TypeError):
                parsed_args = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["function"]["name"],
                    "input": parsed_args,
                }
            )
        return {"role": "assistant", "content": blocks}
