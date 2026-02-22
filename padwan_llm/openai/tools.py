import json
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ..models import ToolCall, ToolCallFunction, ToolDefinition

if TYPE_CHECKING:
    from .types import ChatCompletionTool, CreateChatCompletionStreamResponse

__all__ = ("OpenAIToolMixin",)


class OpenAIToolMixin:
    """Mixin providing tool-calling support for OpenAI-compatible clients and streams."""

    @staticmethod
    def _tools_to_openai(
        tools: Sequence[ToolDefinition],
    ) -> list[ChatCompletionTool]:
        """Convert provider-agnostic ToolDefinitions to OpenAI ChatCompletionTool format."""
        return [
            cast(
                "ChatCompletionTool",
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["parameters"],
                    },
                },
            )
            for t in tools
        ]

    @staticmethod
    def _extract_tool_calls(
        raw_calls: Sequence[dict[str, typing.Any]],
    ) -> list[ToolCall]:
        """Extract shared ToolCall list from an OpenAI-compatible tool_calls response.

        Normalizes Mistral's dict arguments to JSON strings.
        """
        result: list[ToolCall] = []
        for tc in raw_calls:
            func = tc["function"]
            args = func["arguments"]
            arguments = json.dumps(args) if isinstance(args, dict) else args
            result.append(
                ToolCall(
                    id=tc["id"],
                    type="function",
                    function=ToolCallFunction(name=func["name"], arguments=arguments),
                )
            )
        return result

    @staticmethod
    def _accumulate_tool_call_deltas(
        chunk: CreateChatCompletionStreamResponse,
        pending: dict[int, dict[str, str]],
    ) -> None:
        """Accumulate streaming tool call deltas into a pending dict keyed by index.

        Each entry in `pending` tracks partial id, name, and arguments for a
        single tool call being assembled across multiple stream chunks.
        """
        choices = chunk.get("choices")
        if not choices:
            return
        delta = choices[0].get("delta")
        if not delta:
            return
        tool_call_chunks = delta.get("tool_calls")
        if not tool_call_chunks:
            return
        for tc_chunk in tool_call_chunks:
            idx = tc_chunk["index"]
            if idx not in pending:
                pending[idx] = {"id": "", "name": "", "arguments": ""}
            entry = pending[idx]
            if tc_id := tc_chunk.get("id"):
                entry["id"] = tc_id
            if func := tc_chunk.get("function"):
                if name := func.get("name"):
                    entry["name"] = name
                if args := func.get("arguments"):
                    entry["arguments"] += args
