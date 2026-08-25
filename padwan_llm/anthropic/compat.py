from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

from .._json import dumps as _json_dumps
from ..logs import log
from ..openai.client import is_openai_model
from .models import (
    AnthropicCompatBody,
    AnthropicContentBlock,
    AnthropicMessage,
    AnthropicTool,
    AnthropicToolChoice,
)

if TYPE_CHECKING:
    from ..openai.types import (
        ChatCompletionMessageToolCall,
        ChatCompletionRequestAssistantMessage,
        ChatCompletionRequestMessageContentPartImage,
        ChatCompletionRequestMessageContentPartText,
        ChatCompletionRequestSystemMessage,
        ChatCompletionRequestToolMessage,
        ChatCompletionRequestUserMessage,
        ChatCompletionTool,
        CreateChatCompletionRequest,
    )

    type OpenAIRequestMessage = (
        ChatCompletionRequestSystemMessage
        | ChatCompletionRequestUserMessage
        | ChatCompletionRequestAssistantMessage
        | ChatCompletionRequestToolMessage
    )
    type UserContentPart = (
        ChatCompletionRequestMessageContentPartText
        | ChatCompletionRequestMessageContentPartImage
    )

__all__ = ("messages_to_openai",)


def _system_text(system: str | list[AnthropicContentBlock]) -> str | None:
    """Flatten a system prompt (string or text blocks) to a single string."""
    if isinstance(system, str):
        return system or None
    parts = [t for b in system if b.get("type") == "text" and (t := b.get("text"))]
    return "\n\n".join(parts) or None


def _image_source_to_part(
    source: dict[str, Any],
) -> ChatCompletionRequestMessageContentPartImage | None:
    """Convert an Anthropic image `source` to an OpenAI image_url content part."""
    match source.get("type"):
        case "base64":
            media_type = source.get("media_type", "image/png")
            url = f"data:{media_type};base64,{source.get('data', '')}"
        case "url":
            url = source.get("url", "")
        case unknown:
            log.debug("anthropic compat: skipping image source type %r", unknown)
            return None
    return {"type": "image_url", "image_url": {"url": url}}


def _flatten_tool_result_content(
    content: Any,
) -> tuple[str, list[ChatCompletionRequestMessageContentPartImage]]:
    """Flatten a tool_result `content` to text, collecting any image blocks aside.

    OpenAI tool messages carry text only; images found in the result are
    returned separately so the caller can forward them as a user message.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts: list[str] = []
    images: list[ChatCompletionRequestMessageContentPartImage] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            match block.get("type"):
                case "text":
                    if t := block.get("text"):
                        texts.append(t)
                case "image":
                    if part := _image_source_to_part(block.get("source") or {}):
                        images.append(part)
                case unknown:
                    log.debug(
                        "anthropic compat: skipping tool_result block type %r", unknown
                    )
        return "\n".join(texts), images
    return str(content), []


def _user_blocks_to_messages(
    blocks: list[AnthropicContentBlock],
) -> list[OpenAIRequestMessage]:
    """Convert a user message's content blocks, preserving block order.

    tool_result blocks each become a `role: "tool"` message; contiguous
    text/image blocks are grouped into `role: "user"` messages.
    """
    messages: list[OpenAIRequestMessage] = []
    parts: list[UserContentPart] = []

    def flush_parts() -> None:
        if not parts:
            return
        if len(parts) == 1 and parts[0]["type"] == "text":
            content: Any = parts[0]["text"]
        else:
            content = list(parts)
        messages.append({"role": "user", "content": content})
        parts.clear()

    for block in blocks:
        match block.get("type"):
            case "text":
                if text := block.get("text"):
                    parts.append({"type": "text", "text": text})
            case "image":
                if part := _image_source_to_part(block.get("source") or {}):
                    parts.append(part)
            case "tool_result":
                flush_parts()
                text, images = _flatten_tool_result_content(block.get("content"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": text,
                    }
                )
                parts.extend(images)
            case unknown:
                log.debug("anthropic compat: skipping user block type %r", unknown)
    flush_parts()
    return messages


def _assistant_blocks_to_message(
    blocks: list[AnthropicContentBlock],
) -> ChatCompletionRequestAssistantMessage:
    """Convert an assistant message's content blocks to one OpenAI assistant message.

    thinking / redacted_thinking blocks are dropped on replay — their
    signatures only make sense to the Anthropic API.
    """
    texts: list[str] = []
    tool_calls: list[ChatCompletionMessageToolCall] = []
    for block in blocks:
        match block.get("type"):
            case "text":
                if text := block.get("text"):
                    texts.append(text)
            case "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": _json_dumps(block.get("input") or {}),
                        },
                    }
                )
            case "thinking" | "redacted_thinking":
                log.debug("anthropic compat: dropping thinking block on replay")
            case unknown:
                log.debug("anthropic compat: skipping assistant block type %r", unknown)
    message: ChatCompletionRequestAssistantMessage = {"role": "assistant"}
    if texts:
        message["content"] = "\n".join(texts)
    if tool_calls:
        message["tool_calls"] = cast(list, tool_calls)
    return message


def _convert_message(msg: AnthropicMessage) -> list[OpenAIRequestMessage]:
    content = msg["content"]
    if msg["role"] == "assistant":
        if isinstance(content, str):
            return [{"role": "assistant", "content": content}]
        return [_assistant_blocks_to_message(content)]
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    return _user_blocks_to_messages(content)


def _tools_to_openai(tools: list[AnthropicTool]) -> list[ChatCompletionTool]:
    """Convert Anthropic tool definitions, skipping server tools.

    Server tools (API-versioned `type`, no `input_schema`) have no
    OpenAI-compatible equivalent and are dropped.
    """
    converted: list[ChatCompletionTool] = []
    for tool in tools:
        if (schema := tool.get("input_schema")) is None:
            log.debug(
                "anthropic compat: skipping server tool %r (type=%r)",
                tool.get("name"),
                tool.get("type"),
            )
            continue
        function: dict[str, Any] = {"name": tool["name"], "parameters": schema}
        if description := tool.get("description"):
            function["description"] = description
        converted.append(
            cast("ChatCompletionTool", {"type": "function", "function": function})
        )
    return converted


def _apply_tool_choice(
    body: CreateChatCompletionRequest, tool_choice: AnthropicToolChoice
) -> None:
    match tool_choice["type"]:
        case "auto":
            body["tool_choice"] = "auto"
        case "any":
            body["tool_choice"] = "required"
        case "none":
            body["tool_choice"] = "none"
        case "tool":
            body["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.get("name", "")},
            }
    if tool_choice.get("disable_parallel_tool_use"):
        body["parallel_tool_calls"] = False


def messages_to_openai(
    body: AnthropicCompatBody, *, model: str | None = None
) -> CreateChatCompletionRequest:
    """Translate an Anthropic Messages API request to an OpenAI chat completions body.

    `model` overrides the requested (Anthropic) model name with the backend
    model to use; when omitted the original name is kept. `cache_control`
    markers, `metadata`, and the `thinking` config have no OpenAI-compatible
    equivalent and are dropped; thinking blocks are stripped from replayed
    assistant turns.
    """
    target_model = model or body["model"]
    messages: list[OpenAIRequestMessage] = []
    if (system := body.get("system")) is not None:
        if text := _system_text(system):
            messages.append({"role": "system", "content": text})
    for msg in body["messages"]:
        messages.extend(_convert_message(msg))

    request: CreateChatCompletionRequest = {
        "model": target_model,
        "messages": cast(list, messages),
    }
    # Reasoning-tier OpenAI models reject `max_tokens`; other OpenAI-compatible
    # backends may not know `max_completion_tokens` yet.
    max_tokens_key: Literal["max_tokens", "max_completion_tokens"] = (
        "max_completion_tokens" if is_openai_model(target_model) else "max_tokens"
    )
    request[max_tokens_key] = body["max_tokens"]
    if (temperature := body.get("temperature")) is not None:
        request["temperature"] = temperature
    if (top_p := body.get("top_p")) is not None:
        request["top_p"] = top_p
    if body.get("top_k") is not None:
        log.debug("anthropic compat: dropping top_k (no OpenAI equivalent)")
    if stop_sequences := body.get("stop_sequences"):
        request["stop"] = stop_sequences
    if tools := _tools_to_openai(body.get("tools") or []):
        request["tools"] = cast(list, tools)
    if (tool_choice := body.get("tool_choice")) and "tools" in request:
        _apply_tool_choice(request, tool_choice)
    if body.get("thinking"):
        log.debug("anthropic compat: dropping thinking config (backend-specific)")
    return request
