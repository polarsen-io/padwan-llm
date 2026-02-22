import json
import typing
from collections.abc import Sequence

from .models import FunctionDeclaration, GeminiTool
from ..conversation import AssistantToolMessage, ToolResultMessage
from ..models import ToolCall, ToolCallFunction, ToolDefinition

__all__ = ("GeminiToolMixin",)


class GeminiToolMixin:
    """Mixin providing tool-calling support for the Gemini client and stream."""

    @staticmethod
    def _tools_to_gemini(
        tools: Sequence[ToolDefinition],
    ) -> list[GeminiTool]:
        """Convert provider-agnostic ToolDefinitions to Gemini tool format."""
        declarations: list[FunctionDeclaration] = [
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in tools
        ]
        return [{"function_declarations": declarations}]

    @staticmethod
    def _extract_gemini_tool_calls(
        parts: Sequence[dict[str, typing.Any]],
        id_offset: int = 0,
    ) -> list[ToolCall]:
        """Extract ToolCall objects from Gemini response parts containing functionCall.

        Generates synthetic IDs (call_0, call_1, ...) since Gemini doesn't provide them.
        The `id_offset` shifts IDs to avoid collisions when accumulating across stream chunks.
        """
        result: list[ToolCall] = []
        for i, part in enumerate(parts):
            if fc := part.get("functionCall"):
                result.append(
                    ToolCall(
                        id=f"call_{id_offset + i}",
                        type="function",
                        function=ToolCallFunction(
                            name=fc["name"],
                            arguments=json.dumps(fc.get("args", {})),
                        ),
                    )
                )
        return result

    @staticmethod
    def _convert_tool_result(msg: ToolResultMessage) -> dict[str, typing.Any]:
        """Convert a ToolResultMessage to a Gemini functionResponse content dict.

        Gemini expects tool results as a user message with a functionResponse part.
        The content is parsed as JSON if possible, otherwise wrapped in {"result": ...}.
        """
        try:
            response_data = json.loads(msg["content"])
        except (json.JSONDecodeError, TypeError):  # fmt: skip
            response_data = {"result": msg["content"]}
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": msg["name"],
                        "response": response_data,
                    }
                }
            ],
        }

    @staticmethod
    def _convert_assistant_tool_message(
        msg: AssistantToolMessage,
    ) -> dict[str, typing.Any]:
        """Convert an AssistantToolMessage to a Gemini model content dict with functionCall parts.

        Includes a text part if the message has content alongside the tool calls.
        """
        parts: list[dict[str, typing.Any]] = []
        if msg["content"]:
            parts.append({"text": msg["content"]})
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            try:
                parsed_args = json.loads(args)
            except (json.JSONDecodeError, TypeError):  # fmt: skip
                parsed_args = {}
            parts.append(
                {
                    "functionCall": {
                        "name": tc["function"]["name"],
                        "args": parsed_args,
                    }
                }
            )
        return {"role": "model", "parts": parts}
