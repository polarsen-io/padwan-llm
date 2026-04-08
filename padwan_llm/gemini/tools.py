import json
from collections.abc import Sequence
from typing import Any

from .models import (
    Content,
    FunctionCallPart,
    FunctionDeclaration,
    GeminiPart,
    GeminiTool,
    Part,
)
from .._json import loads as _json_loads
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
        parts: Sequence[dict[str, Any]],
        id_offset: int = 0,
    ) -> list[ToolCall]:
        """Extract ToolCall objects from Gemini response parts containing functionCall.

        Generates synthetic IDs (call_0, call_1, ...) since Gemini doesn't provide them.
        The `id_offset` shifts IDs to avoid collisions when accumulating across stream chunks.
        """
        result: list[ToolCall] = []
        for i, part in enumerate(parts):
            if fc := part.get("functionCall"):
                tc = ToolCall(
                    id=fc.get("id") or f"call_{id_offset + i}",
                    type="function",
                    function=ToolCallFunction(
                        name=fc["name"],
                        arguments=json.dumps(fc.get("args", {})),
                    ),
                )
                if sig := part.get("thoughtSignature"):
                    tc["thought_signature"] = sig
                result.append(tc)
        return result

    @staticmethod
    def _convert_tool_result(msg: ToolResultMessage) -> Content:
        """Convert a ToolResultMessage to a Gemini functionResponse content dict.

        Gemini expects tool results as a user message with a functionResponse part.
        The content is parsed as JSON if possible, otherwise wrapped in {"result": ...}.
        """
        try:
            response_data = _json_loads(msg["content"])
        except (json.JSONDecodeError, TypeError):  # fmt: skip
            response_data = {"result": msg["content"]}
        if not isinstance(response_data, dict):
            response_data = {"result": response_data}
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
    ) -> Content:
        """Convert an AssistantToolMessage to a Gemini model content dict with functionCall parts.

        Includes a text part if the message has content alongside the tool calls.
        """
        parts: list[GeminiPart] = []
        if msg["content"]:
            parts.append(Part(text=msg["content"]))
        for tc in msg["tool_calls"]:
            args = tc["function"]["arguments"]
            try:
                parsed_args = _json_loads(args)
            except (json.JSONDecodeError, TypeError):  # fmt: skip
                parsed_args = {}
            part: FunctionCallPart = {
                "functionCall": {
                    "name": tc["function"]["name"],
                    "args": parsed_args,
                    "id": tc["id"],
                }
            }
            if sig := tc.get("thought_signature"):
                part["thoughtSignature"] = sig
            parts.append(part)
        return Content(role="model", parts=parts)
