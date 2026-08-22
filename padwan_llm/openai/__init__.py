# Python 3.15 defers provider components until first use.
__lazy_modules__ = frozenset(
    {
        "padwan_llm.openai.audio",
        "padwan_llm.openai.batch",
        "padwan_llm.openai.client",
        "padwan_llm.openai.realtime",
        "padwan_llm.openai.vision",
    }
)

from .audio import supports_audio
from .batch import BatchJob, BatchRequest, BatchResult
from .client import (
    OPENAI_CHAT_MODELS,
    OPENAI_ENDPOINT,
    OPENAI_MODELS,
    OpenAIClient,
    OpenAIModel,
    is_openai_model,
)
from .realtime import (
    DEFAULT_REALTIME_MODEL,
    NO_TURN_DETECTION,
    REALTIME_ENDPOINT,
    REALTIME_SAMPLE_RATE,
    OpenAIRealtimeClient,
    RealtimeConnection,
    RealtimeServerEvent,
    RealtimeVoice,
)
from .vision import supports_vision

__all__ = (
    "DEFAULT_REALTIME_MODEL",
    "NO_TURN_DETECTION",
    "OPENAI_CHAT_MODELS",
    "OPENAI_ENDPOINT",
    "OPENAI_MODELS",
    "REALTIME_ENDPOINT",
    "REALTIME_SAMPLE_RATE",
    "BatchJob",
    "BatchRequest",
    "BatchResult",
    "OpenAIClient",
    "OpenAIModel",
    "OpenAIRealtimeClient",
    "RealtimeConnection",
    "RealtimeServerEvent",
    "RealtimeVoice",
    "is_openai_model",
    "supports_audio",
    "supports_vision",
)
