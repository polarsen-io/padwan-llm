# Python 3.15 defers provider components until first use.
__lazy_modules__ = frozenset(
    {
        "padwan_llm.mistral.audio",
        "padwan_llm.mistral.client",
        "padwan_llm.mistral.vision",
    }
)

from .audio import supports_audio
from .client import (
    MISTRAL_ENDPOINT,
    MISTRAL_MODELS,
    MistralAudioModel,
    MistralClient,
    MistralEmbeddingModel,
    MistralModel,
    is_mistral_model,
)
from .vision import supports_vision

__all__ = (
    "MISTRAL_ENDPOINT",
    "MISTRAL_MODELS",
    "MistralAudioModel",
    "MistralClient",
    "MistralEmbeddingModel",
    "MistralModel",
    "is_mistral_model",
    "supports_audio",
    "supports_vision",
)
