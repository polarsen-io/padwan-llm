from .mistral import is_mistral_model
from .openai import is_openai_model

__all__ = ("supports_vision",)

# Known OpenAI models that do NOT accept image input. Everything else that looks
# like an OpenAI model is assumed multimodal (gpt-4o/4.1/5 families, gpt-4-turbo,
# o1, o3, o4-mini all take images).
_OPENAI_TEXT_ONLY = frozenset(
    {"gpt-4", "o1-mini", "o1-preview", "o3-mini", "codex-mini-latest"}
)
# Text-only OpenAI-compatible families, matched by prefix because they ship many
# size/date variants: open-weight gpt-oss (gpt-oss-20b/120b) and legacy gpt-3.5.
_OPENAI_TEXT_ONLY_PREFIXES = ("gpt-oss", "gpt-3.5", "gpt-35")


def supports_vision(model: str | None) -> bool:
    """Best-effort check of whether a model accepts image input.

    Detection is curated, not authoritative: Mistral is text-only except the
    `pixtral-` line, OpenAI excludes a small set of known text-only models and
    families (gpt-oss, gpt-3.5), and Gemini/Grok plus any unknown (e.g. local)
    model default to ``True`` so the caller attempts the request rather than
    blocking — the provider surfaces a clear error if it really cannot see images.
    """
    if not model:
        return False
    if is_mistral_model(model):
        return model.startswith("pixtral-")
    if is_openai_model(model):
        if model.startswith(_OPENAI_TEXT_ONLY_PREFIXES):
            return False
        return model not in _OPENAI_TEXT_ONLY
    # Gemini, Grok, and unknown/local endpoints: assume multimodal-capable.
    return True
