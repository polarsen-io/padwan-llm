from importlib.metadata import version as _pkg_version

from ._base import ChatStream, LLMClientBase, OnThought
from ._deprecation import ModelDeprecationWarning
from .agent import AgentSession, ConversationStore, OnMcpConnect, ToolCallContext
from .anthropic import (
    ANTHROPIC_MODELS,
    AnthropicClient,
    AnthropicModel,
    is_anthropic_model,
)
from .client import LLMClient
from .content import (
    ContentImagePart,
    ContentPart,
    ContentTextPart,
    content_parts,
    image_part,
    text_file_part,
    text_part,
)
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
from .mcp import McpStdio, McpStreamable, McpTool, McpTransport, OnAuth, ProgressEvent
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
    RealtimeClient,
    RealtimeConnection,
    RealtimeServerEvent,
    is_openai_model,
)
from .vision import supports_vision

__all__ = (
    "AgentSession",
    "AnthropicClient",
    "AnthropicModel",
    "AssistantToolMessage",
    "ChatMessage",
    "ChatResponse",
    "ChatStream",
    "ContentImagePart",
    "ContentPart",
    "ContentTextPart",
    "ConversationSnapshot",
    "ConversationState",
    "ConversationStore",
    "FinishReason",
    "content_parts",
    "image_part",
    "supports_vision",
    "text_file_part",
    "text_part",
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
    "ModelDeprecationWarning",
    "OnAuth",
    "OnMcpConnect",
    "OnThought",
    "ProgressEvent",
    "Message",
    "MistralClient",
    "MistralModel",
    "OpenAIClient",
    "OpenAIModel",
    "RealtimeClient",
    "RealtimeConnection",
    "RealtimeServerEvent",
    "Provider",
    "ToolCallContext",
    "ToolCall",
    "ToolCallFunction",
    "ToolDefinition",
    "ToolResultMessage",
    "UsageToken",
    "ANTHROPIC_MODELS",
    "GEMINI_MODELS",
    "GROK_MODELS",
    "MISTRAL_MODELS",
    "OPENAI_MODELS",
    "OPENAI_CHAT_MODELS",
    "is_anthropic_model",
    "is_gemini_model",
    "is_grok_model",
    "is_mistral_model",
    "is_openai_model",
    "__version__",
)

__version__: str = _pkg_version("padwan-llm")
