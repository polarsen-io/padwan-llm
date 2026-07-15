from typing import NotRequired, TypedDict, overload

from ._base import LLMClientBase, OnThought
from .anthropic import AnthropicClient, AnthropicModel, is_anthropic_model
from .gemini import GeminiClient, GeminiModel, is_gemini_model
from .grok import GrokClient, GrokModel, is_grok_model
from .mistral import MistralClient, MistralModel, is_mistral_model
from .openai import OpenAIClient, OpenAIModel, is_openai_model

__all__ = ("LLMClient",)


class _ClientKwargs(TypedDict):
    model: str
    temperature: float
    timeout: float
    api_key: str | None
    on_thought: OnThought | None
    base_url: NotRequired[str]


@overload
def LLMClient(
    model: OpenAIModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> OpenAIClient: ...
@overload
def LLMClient(
    model: GeminiModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> GeminiClient: ...
@overload
def LLMClient(
    model: MistralModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> MistralClient: ...
@overload
def LLMClient(
    model: GrokModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> GrokClient: ...
@overload
def LLMClient(
    model: AnthropicModel,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> AnthropicClient: ...
@overload
def LLMClient(
    model: str,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> LLMClientBase: ...


def LLMClient(
    model: str,
    *,
    temperature: float = 0.2,
    timeout: float = 60,
    api_key: str | None = None,
    on_thought: OnThought | None = None,
    base_url: str | None = None,
) -> LLMClientBase:
    """Create an LLM client based on model name.

    When ``base_url`` is provided it overrides the provider's default
    endpoint, so you can point any provider at a custom deployment
    (e.g. a self-hosted Mistral on a private URL).  For unknown models
    the ``api_key`` defaults to ``"no-key-required"`` to support local
    inference servers (llama.cpp, Ollama, vLLM, etc.).

    The ``on_thought`` callback, when provided, receives reasoning/thinking
    chunks from providers that support them (Gemini, Grok, Mistral).
    Providers that don't emit thoughts simply never invoke it.
    """
    kwargs = _ClientKwargs(
        model=model,
        temperature=temperature,
        timeout=timeout,
        api_key=api_key,
        on_thought=on_thought,
    )
    if base_url is not None:
        kwargs["base_url"] = base_url
    if is_openai_model(model):
        return OpenAIClient(**kwargs)
    if is_gemini_model(model):
        return GeminiClient(**kwargs)
    if is_mistral_model(model):
        return MistralClient(**kwargs)
    if is_grok_model(model):
        return GrokClient(**kwargs)
    if is_anthropic_model(model):
        return AnthropicClient(**kwargs)
    if kwargs["api_key"] is None:
        kwargs["api_key"] = "no-key-required"
    return OpenAIClient(**kwargs)
