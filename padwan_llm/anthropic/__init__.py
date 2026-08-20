# Python 3.15 defers provider components until first use.
__lazy_modules__ = frozenset(
    {"padwan_llm.anthropic.client", "padwan_llm.anthropic.vision"}
)

from .client import (
    ANTHROPIC_ENDPOINT,
    ANTHROPIC_MODELS,
    AnthropicChatStream,
    AnthropicClient,
    AnthropicModel,
    is_anthropic_model,
)
from .vision import supports_vision

__all__ = (
    "ANTHROPIC_ENDPOINT",
    "ANTHROPIC_MODELS",
    "AnthropicChatStream",
    "AnthropicClient",
    "AnthropicModel",
    "is_anthropic_model",
    "supports_vision",
)
