from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

from .._base import env_api_key
from ..errors import Provider
from ..openai.realtime import OpenAIRealtimeClient

__all__ = (
    "VOICE_ENDPOINT",
    "DEFAULT_VOICE_MODEL",
    "GrokVoice",
    "GrokVoiceModel",
    "GrokRealtimeClient",
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
class GrokRealtimeClient(OpenAIRealtimeClient):
    """Speech-to-speech client for the Grok Voice Agent API.

    The wire protocol is OpenAI Realtime-compatible, so this only rebinds the
    endpoint, model, voice, and key resolution (``GROK_API_KEY``). Grok
    transcribes natively, so *transcription_model* defaults to ``None``.
    """

    provider: ClassVar[Provider] = "grok"

    model: str = DEFAULT_VOICE_MODEL
    base_url: str = VOICE_ENDPOINT
    voice: GrokVoice = "eve"
    transcription_model: str | None = None

    def _get_default_api_key(self) -> str:
        return env_api_key(self.provider, "GROK_API_KEY")
