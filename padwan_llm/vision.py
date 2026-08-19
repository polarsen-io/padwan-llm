from .anthropic import is_anthropic_model, supports_vision as _anthropic_vision
from .gemini import is_gemini_model, supports_vision as _gemini_vision
from .grok import is_grok_model, supports_vision as _grok_vision
from .mistral import is_mistral_model, supports_vision as _mistral_vision
from .openai import is_openai_model, supports_vision as _openai_vision

__all__ = ("supports_vision",)


def supports_vision(model: str | None) -> bool:
    """Best-effort check of whether a model accepts image input.

    Dispatches to the provider's curated ``supports_vision`` (same routing order
    as ``LLMClient``). Unknown (e.g. local) models default to ``True`` so the
    caller attempts the request rather than blocking — the provider surfaces a
    clear error if it really cannot see images.
    """
    if not model:
        return False
    if is_openai_model(model):
        return _openai_vision(model)
    if is_gemini_model(model):
        return _gemini_vision(model)
    if is_mistral_model(model):
        return _mistral_vision(model)
    if is_grok_model(model):
        return _grok_vision(model)
    if is_anthropic_model(model):
        return _anthropic_vision(model)
    return True
