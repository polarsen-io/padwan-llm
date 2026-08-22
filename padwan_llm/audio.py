from .anthropic import is_anthropic_model, supports_audio as _anthropic_audio
from .gemini import is_gemini_model, supports_audio as _gemini_audio
from .grok import is_grok_model, supports_audio as _grok_audio
from .mistral import is_mistral_model, supports_audio as _mistral_audio
from .openai import is_openai_model, supports_audio as _openai_audio

__all__ = ("supports_audio",)


def supports_audio(model: str | None) -> bool:
    """Best-effort check of whether a model accepts audio input.

    Dispatches to the provider's curated ``supports_audio`` (same routing order
    as ``LLMClient``). Unknown (e.g. local) models default to ``True`` so the
    caller attempts the request rather than blocking (the provider surfaces a
    clear error if it really cannot hear audio).
    """
    if not model:
        return False
    if is_openai_model(model):
        return _openai_audio(model)
    if is_gemini_model(model):
        return _gemini_audio(model)
    if is_mistral_model(model):
        return _mistral_audio(model)
    if is_grok_model(model):
        return _grok_audio(model)
    if is_anthropic_model(model):
        return _anthropic_audio(model)
    return True
