from typing import overload

from ._base import LLMClientBase, OnThought
from .gemini import GeminiClient, GeminiModel, is_gemini_model
from .grok import GrokClient, GrokModel, is_grok_model
from .mistral import MistralClient, MistralModel, is_mistral_model
from .openai import OpenAIClient, OpenAIModel, is_openai_model

__all__ = ("LLMClient",)


@overload
def LLMClient(
    model: OpenAIModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
) -> OpenAIClient: ...
@overload
def LLMClient(
    model: GeminiModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
) -> GeminiClient: ...
@overload
def LLMClient(
    model: MistralModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
) -> MistralClient: ...
@overload
def LLMClient(
    model: GrokModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
) -> GrokClient: ...
@overload
def LLMClient(
    model: str,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
) -> LLMClientBase: ...


def LLMClient(
    model: str,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
) -> LLMClientBase:
    """Create an LLM client based on model name.

    The ``on_thought`` callback, when provided, receives reasoning/thinking
    chunks from providers that support them (Gemini, Grok, Mistral).
    Providers that don't emit thoughts simply never invoke it.
    """
    kwargs = {
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
        "api_key": api_key,
        "on_thought": on_thought,
    }
    if is_openai_model(model):
        return OpenAIClient(**kwargs)
    if is_gemini_model(model):
        return GeminiClient(**kwargs)
    if is_mistral_model(model):
        return MistralClient(**kwargs)
    if is_grok_model(model):
        return GrokClient(**kwargs)
    raise ValueError(f"Unknown model: {model}")
