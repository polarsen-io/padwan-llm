from dataclasses import dataclass
from typing import Literal, get_args

from ..openai.realtime import OpenAIRealtimeClient
from .client import _GrokAuth

__all__ = (
    "DEFAULT_VOICE_MODEL",
    "VOICE_ENDPOINT",
    "GrokRealtimeClient",
    "GrokVoice",
    "GrokVoiceModel",
)

VOICE_ENDPOINT = "wss://api.x.ai/v1/realtime"
DEFAULT_VOICE_MODEL = "grok-voice-latest"

GrokVoiceModel = Literal[
    "grok-voice-latest",
    "grok-voice-think-fast-2.0",
    "grok-voice-think-fast-1.0",
]

_KnownGrokVoice = Literal["eve", "ara", "leo"]
# `| str` keeps arbitrary (e.g. cloned) voices valid while the Literal member
# drives IDE completion.
type GrokVoice = _KnownGrokVoice | str
_GROK_VOICES: tuple[str, ...] = get_args(_KnownGrokVoice)


@dataclass
class GrokRealtimeClient(_GrokAuth, OpenAIRealtimeClient):
    """Speech-to-speech client for the Grok Voice Agent API.

    The wire protocol is OpenAI Realtime-compatible, so this only rebinds the
    endpoint, model, voice, and key resolution (``GROK_API_KEY``). Grok
    transcribes natively, so *transcription_model* defaults to ``None``.
    """

    model: str = DEFAULT_VOICE_MODEL
    base_url: str = VOICE_ENDPOINT
    voice: GrokVoice = "eve"
    transcription_model: str | None = None
