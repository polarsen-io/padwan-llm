from importlib.metadata import version as _pkg_version

from ._base import ChatStream, LLMClientBase, OnThought
from .agent import AgentSession, ConversationStore, OnMcpConnect, ToolCallContext
from .mcp import McpStreamable, McpStdio, McpTool, McpTransport, OnAuth, ProgressEvent
from .client import LLMClient
from .conversation import (
    AssistantToolMessage,
    ChatMessage,
    ConversationSnapshot,
    ConversationState,
    Message,
    ToolResultMessage,
)
from .errors import LLMError, Provider
from .gemini import GEMINI_MODELS, GeminiClient, GeminiModel, is_gemini_model
from .grok import GROK_MODELS, GrokClient, GrokModel, is_grok_model
from .mistral import MISTRAL_MODELS, MistralClient, MistralModel, is_mistral_model
from .models import (
    ChatResponse,
    FinishReason,
    ToolCall,
    ToolCallFunction,
    ToolDefinition,
    UsageToken,
)
from .openai import (
    OPENAI_CHAT_MODELS,
    OPENAI_MODELS,
    OpenAIClient,
    OpenAIModel,
    is_openai_model,
)

__all__ = (
    "AgentSession",
    "AssistantToolMessage",
    "ChatMessage",
    "ChatResponse",
    "ChatStream",
    "ConversationSnapshot",
    "ConversationState",
    "ConversationStore",
    "FinishReason",
    "GeminiClient",
    "GeminiModel",
    "GrokClient",
    "GrokModel",
    "LLMClient",
    "LLMClientBase",
    "LLMError",
    "McpStreamable",
    "McpStdio",
    "McpTool",
    "McpTransport",
    "OnAuth",
    "OnMcpConnect",
    "OnThought",
    "ProgressEvent",
    "Message",
    "MistralClient",
    "MistralModel",
    "OpenAIClient",
    "OpenAIModel",
    "Provider",
    "ToolCallContext",
    "ToolCall",
    "ToolCallFunction",
    "ToolDefinition",
    "ToolResultMessage",
    "UsageToken",
    "GEMINI_MODELS",
    "GROK_MODELS",
    "MISTRAL_MODELS",
    "OPENAI_MODELS",
    "OPENAI_CHAT_MODELS",
    "is_gemini_model",
    "is_grok_model",
    "is_mistral_model",
    "is_openai_model",
    "__version__",
)

__version__: str = _pkg_version("padwan-llm")
