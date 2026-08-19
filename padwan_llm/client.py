import os
from collections.abc import Mapping, Sequence
from typing import Any, NotRequired, TypedDict, overload

from ._base import LLMClientBase, OnThought
from .anthropic import AnthropicClient, AnthropicModel, is_anthropic_model
from .gemini import GeminiClient, GeminiModel, GeminiRealtimeClient, is_gemini_model
from .gemini.realtime import GeminiLiveModel
from .grok import GrokClient, GrokModel, GrokRealtimeClient, is_grok_model
from .grok.realtime import GrokVoiceModel
from .mistral import MistralClient, MistralModel, is_mistral_model
from .openai import (
    DEFAULT_REALTIME_MODEL,
    REALTIME_SAMPLE_RATE,
    OpenAIClient,
    OpenAIModel,
    OpenAIRealtimeClient,
    is_openai_model,
)

__all__ = ("LLMClient", "RealtimeClient")

# Unified gateway: a single OpenAI-compatible endpoint + token serving every
# model family. Set these to route all models through one endpoint/token
# instead of per-provider env vars.
PADWAN_BASE_URL_ENV = "PADWAN_BASE_URL"
PADWAN_API_KEY_ENV = "PADWAN_API_KEY"


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

    Setting ``PADWAN_BASE_URL`` in the environment switches on *gateway
    mode*: a single OpenAI-compatible endpoint and token (``PADWAN_API_KEY``)
    serve every model, so an aggregator that exposes OSS variants of many
    families behind one URL works without per-provider env vars or per-call
    overrides. In gateway mode all models route through ``OpenAIClient``
    regardless of name prefix; an explicit ``base_url`` argument disables it
    and restores native per-provider routing. An explicit ``base_url`` still
    authenticates with ``PADWAN_API_KEY`` when it is set and no ``api_key``
    is given, so provider keys are never sent to a custom endpoint.

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
    gateway_url = os.environ.get(PADWAN_BASE_URL_ENV)
    if base_url is None and gateway_url:
        kwargs["base_url"] = gateway_url
        # Scope the token to the gateway: never fall through to OPENAI_API_KEY.
        kwargs["api_key"] = (
            api_key or os.environ.get(PADWAN_API_KEY_ENV) or "no-key-required"
        )
        return OpenAIClient(**kwargs)
    if base_url is not None:
        kwargs["base_url"] = base_url
        # Explicit custom endpoint: prefer the gateway token over provider env
        # keys so a provider secret is never sent to a third-party endpoint.
        if api_key is None and (padwan_key := os.environ.get(PADWAN_API_KEY_ENV)):
            kwargs["api_key"] = padwan_key
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


# The literal overloads intentionally overlap the str fallback (which assumes
# OpenAI, mirroring the runtime prefix dispatch), hence the overlap ignore.
@overload
def RealtimeClient(  # pyright: ignore[reportOverlappingOverload]
    model: GeminiModel | GeminiLiveModel,
    *,
    instructions: str | None = None,
    voice: str | None = None,
    turn_detection: Mapping[str, Any] | str | None = None,
    transcription_model: str | None = "whisper-1",
    output_modalities: Sequence[str] = ("audio",),
    sample_rate: int = REALTIME_SAMPLE_RATE,
    timeout: float = 30.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> GeminiRealtimeClient: ...
@overload
def RealtimeClient(
    model: GrokModel | GrokVoiceModel,
    *,
    instructions: str | None = None,
    voice: str | None = None,
    turn_detection: Mapping[str, Any] | str | None = None,
    transcription_model: str | None = "whisper-1",
    output_modalities: Sequence[str] = ("audio",),
    sample_rate: int = REALTIME_SAMPLE_RATE,
    timeout: float = 30.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> GrokRealtimeClient: ...
@overload
def RealtimeClient(
    model: str = DEFAULT_REALTIME_MODEL,
    *,
    instructions: str | None = None,
    voice: str | None = None,
    turn_detection: Mapping[str, Any] | str | None = None,
    transcription_model: str | None = "whisper-1",
    output_modalities: Sequence[str] = ("audio",),
    sample_rate: int = REALTIME_SAMPLE_RATE,
    timeout: float = 30.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAIRealtimeClient: ...


def RealtimeClient(
    model: str = DEFAULT_REALTIME_MODEL,
    *,
    instructions: str | None = None,
    voice: str | None = None,
    turn_detection: Mapping[str, Any] | str | None = None,
    transcription_model: str | None = "whisper-1",
    output_modalities: Sequence[str] = ("audio",),
    sample_rate: int = REALTIME_SAMPLE_RATE,
    timeout: float = 30.0,
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAIRealtimeClient | GeminiRealtimeClient:
    """Create a realtime speech-to-speech client based on model name.

    ``async with RealtimeClient(...) as conn:`` yields the live connection.
    Gemini models route to :class:`GeminiRealtimeClient` (Live API), Grok
    models to :class:`GrokRealtimeClient` (OpenAI Realtime-compatible), and
    everything else to :class:`OpenAIRealtimeClient`. *voice* ``None`` uses
    the provider default. When no *api_key* is given, an explicit *base_url*
    prefers ``PADWAN_API_KEY`` over the provider env key (which remains the
    fallback).
    """
    if base_url is not None and api_key is None:
        api_key = os.environ.get(PADWAN_API_KEY_ENV)
    client: OpenAIRealtimeClient | GeminiRealtimeClient
    if is_gemini_model(model):
        if sample_rate != REALTIME_SAMPLE_RATE:
            raise ValueError(
                "sample_rate is fixed for the Gemini Live API (16 kHz in, 24 kHz out)"
            )
        client = GeminiRealtimeClient(
            model=model,
            instructions=instructions,
            turn_detection=turn_detection,
            transcription=transcription_model is not None,
            output_modalities=output_modalities,
            api_key=api_key,
            timeout=timeout,
        )
    else:
        client_cls = (
            GrokRealtimeClient if is_grok_model(model) else OpenAIRealtimeClient
        )
        client = client_cls(
            model=model,
            instructions=instructions,
            turn_detection=turn_detection,
            output_modalities=output_modalities,
            sample_rate=sample_rate,
            api_key=api_key,
            timeout=timeout,
        )
        # Grok transcribes natively; only forward the OpenAI-style default there
        # when explicitly set.
        if not is_grok_model(model) or transcription_model != "whisper-1":
            client.transcription_model = transcription_model
    if voice is not None:
        client.voice = voice
    if base_url is not None:
        client.base_url = base_url
    return client
